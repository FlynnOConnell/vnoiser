"""The ``PF`` folder: what the spatial JEDI pipeline saved per experiment.

``<animal>/<experiment>/PF/`` is what :class:`~vnoiser.dataset.SpatialJediDataset`
and the curation dashboard read. :class:`PfWriter` writes every file in the
archive's shapes; :func:`read_pf` reads them back through the same restricted
unpickler the dataset uses, so a written folder is known to open.

Files and shapes (scan ids are strings, ``"35"``):

- ``denoised_trace_scans.pkl``: ``{scan: {domain: float64 (T,)}}``, the
  trace the dashboard curates (:func:`final_trace`). Domain names here are
  the final ones (``soma1`` -> ``soma``).
- ``fs_scans.pkl``: ``{scan: int}``, the frame rate rounded down.
- ``scanIDs_ROIs.pkl``: ``scanID_spatial``, ``scanID_1st_env``,
  ``domain_ROInumber`` (with ``All_domains``), ``roi_list``.
- ``detected_events_peaks.pkl``: ``{scan: {domain: int64 (n,)}}``.
- ``param_spike_detect.pkl``: the :class:`~vnoiser.events.SpikeDetectConfig`
  thresholds.
- ``denoised_trace_components.pkl``: ``{scan: {domain: {rescaled_signal (1, T)
  complex, lp_FIR1Hz, lp_FIR100Hz, envelope_lp}}}`` with the pipeline's
  domain names.
- ``test.h5``: ``<scan>/dfof_raw`` and ``<scan>/dfof_zscore``, ``(D, T)``,
  rows in ``domain_ROInumber`` order without ``All_domains`` / ``bg``.
- ``cwts.h5`` (optional, large): ``<scan>/cwt_dfof (S, T, D) complex64``,
  ``cwt_freq (S, D)``, ``freq_scale (S,)``.
- ``pipeline.json``: provenance, ours only.
"""

from __future__ import annotations

import collections
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence

import h5py
import numpy as np

from .dataset import SpatialJediDataset, _RestrictedUnpickler, _restricted_pickle_load
from .preprocess import EXCLUDED_DOMAINS, domain_names

TRACES_FILE = SpatialJediDataset.trace_filename
FS_FILE = "fs_scans.pkl"
ROIS_FILE = "scanIDs_ROIs.pkl"
PEAKS_FILE = "detected_events_peaks.pkl"
PARAMS_FILE = "param_spike_detect.pkl"
COMPONENTS_FILE = "denoised_trace_components.pkl"
DFOF_FILE = "test.h5"
CWT_FILE = "cwts.h5"
PROVENANCE_FILE = "pipeline.json"
PIPELINE_FILES = (
    TRACES_FILE, FS_FILE, ROIS_FILE, PEAKS_FILE, PARAMS_FILE, COMPONENTS_FILE,
    DFOF_FILE, CWT_FILE, PROVENANCE_FILE,
)
# the archive renamed one domain between the components and the final traces
FINAL_DOMAIN_NAMES = {"soma1": "soma"}


def _nested_dict():
    """The notebook-local factory the archive's components pickle refers to."""
    return collections.defaultdict(_nested_dict)


class _ComponentsUnpickler(_RestrictedUnpickler):
    """The restricted unpickler plus the archive's ``__main__.nested_dict``."""

    def find_class(self, module, name):
        if name == "nested_dict":
            return _nested_dict
        return super().find_class(module, name)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def final_domain_name(name: str) -> str:
    """The domain's key in ``denoised_trace_scans.pkl`` and the peaks file."""
    return FINAL_DOMAIN_NAMES.get(str(name), str(name))


def final_trace(rescaled_signal, lp_fir1hz) -> np.ndarray:
    """The curated trace: the real part of the masked wavelet sum, shifted by
    the first sample of the 1 Hz baseline.

    That constant is what the archive's ``denoised_trace_scans.pkl`` carries
    over ``real(rescaled_signal)`` (checked on every trace of
    ``stan112_expt12``); the baseline itself is not added.
    """
    rescaled_signal = np.asarray(rescaled_signal).ravel()
    lp_fir1hz = np.asarray(lp_fir1hz, dtype=float).ravel()
    return np.real(rescaled_signal) + float(lp_fir1hz[0])


@dataclass
class DomainResult:
    """One domain of one scan after denoising.

    ``rescaled_signal`` is complex when the reducer kept complex bands;
    ``peaks`` is None when detection was skipped; ``cwt`` holds
    ``(coefficients (S, T), frequencies (S,))`` only when kept for saving;
    ``timing`` holds the wall seconds of each stage that produced it.
    """

    rescaled_signal: np.ndarray
    lp_fir1hz: np.ndarray
    lp_fir100hz: np.ndarray
    envelope_lp: np.ndarray
    peaks: np.ndarray | None = None
    cwt: tuple | None = None
    timing: dict | None = None

    @property
    def trace(self) -> np.ndarray:
        return final_trace(self.rescaled_signal, self.lp_fir1hz)


class PfWriter:
    """Writes one experiment's ``PF`` folder domain by domain.

    Parameters
    ----------
    pf_dir : path
        The folder to create (``<animal>/<experiment>/PF``).
    domains : mapping of domain name -> ROI indices
        ``domain_ROInumber``; ``All_domains`` is added when missing.
    scan_ids : sequence of str
        ``scanID_spatial``, the scans in order.
    first_env : sequence of str, optional
        ``scanID_1st_env``: the first scan of each environment.
    roi_list : mapping of scan id -> ROI indices, optional
        Every ROI of each scan; defaults to the union of ``domains``.
    spike_params : dict, optional
        ``param_spike_detect.pkl`` payload
        (:meth:`~vnoiser.events.SpikeDetectConfig.to_param_pickle`).
    save_cwt : bool, default False
        Also write ``cwts.h5`` from ``DomainResult.cwt``. Large.
    freq_scales : array, optional
        The CWT scales, for ``cwts.h5``.
    provenance : dict, optional
        Extra keys for ``pipeline.json``.
    overwrite : bool, default False
        Replace an existing folder's files. Otherwise an existing
        ``denoised_trace_scans.pkl`` raises.

    Call :meth:`add_scan` once per scan, :meth:`add_domain` per domain, then
    :meth:`finish` (or use it as a context manager).
    """

    def __init__(
        self,
        pf_dir,
        *,
        domains: Mapping[str, Sequence[int]],
        scan_ids: Sequence[str],
        first_env: Sequence[str] = (),
        roi_list: Mapping[str, Sequence[int]] | None = None,
        spike_params: dict | None = None,
        save_cwt: bool = False,
        freq_scales=None,
        provenance: dict | None = None,
        overwrite: bool = False,
    ):
        self.pf_dir = Path(pf_dir)
        self.domains = {str(k): [int(v) for v in rois] for k, rois in domains.items()}
        self.names = domain_names(self.domains, EXCLUDED_DOMAINS)
        if not self.names:
            raise ValueError("no domains to write")
        self.scan_ids = [str(s) for s in scan_ids]
        self.first_env = [str(s) for s in first_env]
        self.roi_list = (
            {str(k): sorted(int(v) for v in rois) for k, rois in roi_list.items()}
            if roi_list is not None else None
        )
        self.spike_params = dict(spike_params) if spike_params else None
        self.save_cwt = bool(save_cwt)
        self.freq_scales = None if freq_scales is None else np.asarray(freq_scales, dtype=float)
        self.provenance = dict(provenance or {})
        self.overwrite = bool(overwrite)
        self._fs: dict = {}
        self._traces: dict = {}
        self._components: dict = {}
        self._peaks: dict = {}
        self._n_frames: dict = {}
        self._finished = False
        if (self.pf_dir / TRACES_FILE).exists() and not self.overwrite:
            raise FileExistsError(f"{self.pf_dir / TRACES_FILE} exists; pass overwrite=True")
        self.pf_dir.mkdir(parents=True, exist_ok=True)
        # a rerun replaces every pipeline file so nothing stale survives; .curation stays
        for name in PIPELINE_FILES:
            path = self.pf_dir / name
            if path.exists():
                path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()

    def add_scan(self, scan_id, fs_hz: float, dfof_raw, z) -> None:
        """Record a scan's frame rate and write its dF/F and z rows."""
        scan_id = str(scan_id)
        if scan_id not in self.scan_ids:
            raise KeyError(f"scan {scan_id} is not in scan_ids {self.scan_ids}")
        dfof_raw = np.asarray(dfof_raw, dtype=float)
        z = np.asarray(z, dtype=float)
        if dfof_raw.shape != z.shape or dfof_raw.shape[0] != len(self.names):
            raise ValueError(
                f"dfof_raw {dfof_raw.shape} and z {z.shape} must be ({len(self.names)}, n_frame)"
            )
        self._fs[scan_id] = float(fs_hz)
        self._n_frames[scan_id] = int(z.shape[1])
        with h5py.File(self.pf_dir / DFOF_FILE, "a") as f:
            if scan_id in f:
                del f[scan_id]
            grp = f.create_group(scan_id)
            grp.create_dataset("dfof_raw", data=dfof_raw, compression="gzip", compression_opts=9)
            grp.create_dataset("dfof_zscore", data=z, compression="gzip", compression_opts=9)

    def add_domain(self, scan_id, domain, result: DomainResult) -> None:
        """Record one domain's result; ``add_scan`` must have come first."""
        scan_id, domain = str(scan_id), str(domain)
        if scan_id not in self._fs:
            raise KeyError(f"add_scan({scan_id!r}, ...) first")
        if domain not in self.names:
            raise KeyError(f"{domain} is not one of {self.names}")
        n = self._n_frames[scan_id]
        trace = result.trace
        if trace.size != n:
            raise ValueError(f"{scan_id}/{domain}: {trace.size} samples, scan has {n}")
        self._traces.setdefault(scan_id, {})[final_domain_name(domain)] = trace.astype(float)
        self._components.setdefault(scan_id, {})[domain] = {
            "rescaled_signal": np.asarray(result.rescaled_signal).reshape(1, -1),
            "lp_FIR1Hz": np.asarray(result.lp_fir1hz, dtype=float),
            "lp_FIR100Hz": np.asarray(result.lp_fir100hz, dtype=float),
            "envelope_lp": np.asarray(result.envelope_lp, dtype=float),
        }
        if result.peaks is not None:
            self._peaks.setdefault(scan_id, {})[final_domain_name(domain)] = (
                np.asarray(result.peaks, dtype=np.int64).ravel()
            )
        if not self.save_cwt:
            return
        if result.cwt is None:
            raise ValueError(f"{scan_id}/{domain}: save_cwt needs DomainResult.cwt")
        coeff = np.asarray(result.cwt[0])
        freqs = np.asarray(result.cwt[1], dtype=float)
        n_scales = coeff.shape[0]
        row = self.names.index(domain)
        with h5py.File(self.pf_dir / CWT_FILE, "a") as f:
            if scan_id not in f:
                grp = f.create_group(scan_id)
                grp.create_dataset(
                    "cwt_dfof", shape=(n_scales, n, len(self.names)), dtype=np.complex64,
                    chunks=(n_scales, min(n, 4096), 1), compression="gzip", compression_opts=4,
                )
                grp.create_dataset("cwt_freq", shape=(n_scales, len(self.names)), dtype=float)
                scales = self.freq_scales if self.freq_scales is not None else np.full(n_scales, np.nan)
                grp.create_dataset("freq_scale", data=np.asarray(scales, dtype=float))
            f[scan_id]["cwt_dfof"][:, :, row] = coeff.astype(np.complex64)
            f[scan_id]["cwt_freq"][:, row] = freqs

    def finish(self) -> dict:
        """Write the pickles and the provenance file; returns ``{name: path}``."""
        if not self._finished:
            missing = [s for s in self.scan_ids if s not in self._traces]
            if missing:
                raise ValueError(f"no domains were added for scan(s) {missing}")
            for scan_id, traces in self._traces.items():
                absent = [final_domain_name(d) for d in self.names if final_domain_name(d) not in traces]
                if absent:
                    raise ValueError(f"scan {scan_id} is missing domain(s) {absent}")
            rois = set()
            for name in self.names:
                rois.update(self.domains[name])
            if self.roi_list:
                for values in self.roi_list.values():
                    rois.update(values)
            rois = sorted(rois)
            domain_rois = {"All_domains": list(range(max(rois) + 1))} if rois else {}
            domain_rois.update({k: list(v) for k, v in self.domains.items() if k != "All_domains"})
            if "All_domains" in self.domains:
                domain_rois["All_domains"] = list(self.domains["All_domains"])
            roi_list = self.roi_list or {s: list(rois) for s in self.scan_ids}
            payloads = {
                TRACES_FILE: {s: dict(self._traces[s]) for s in self.scan_ids},
                FS_FILE: {s: int(np.floor(self._fs[s])) for s in self.scan_ids},
                ROIS_FILE: {
                    "scanID_spatial": list(self.scan_ids),
                    "scanID_1st_env": list(self.first_env),
                    "domain_ROInumber": domain_rois,
                    "roi_list": {s: list(roi_list[s]) for s in self.scan_ids},
                },
                COMPONENTS_FILE: {s: dict(self._components[s]) for s in self.scan_ids},
            }
            if self._peaks:
                payloads[PEAKS_FILE] = {s: dict(self._peaks.get(s, {})) for s in self.scan_ids}
            if self.spike_params:
                payloads[PARAMS_FILE] = dict(self.spike_params)
            for name, payload in payloads.items():
                with (self.pf_dir / name).open("wb") as handle:
                    pickle.dump(payload, handle)
            provenance = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scan_ids": list(self.scan_ids),
                "first_env": list(self.first_env),
                "domains": domain_rois,
                "fs_hz": {s: self._fs[s] for s in self.scan_ids},
                "n_frames": {s: self._n_frames[s] for s in self.scan_ids},
                "files": sorted(payloads) + [DFOF_FILE] + ([CWT_FILE] if self.save_cwt else []),
            }
            provenance.update(self.provenance)
            (self.pf_dir / PROVENANCE_FILE).write_text(
                json.dumps(provenance, indent=2, default=_json_default)
            )
            self._finished = True
        return {p.name: p for p in sorted(self.pf_dir.iterdir()) if p.is_file()}


@dataclass
class PfFiles:
    """One ``PF`` folder read back."""

    pf_dir: Path
    traces: dict
    fs: dict
    rois: dict
    peaks: dict | None = None
    params: dict | None = None
    components: dict | None = None
    provenance: dict | None = None

    @property
    def scan_ids(self) -> list:
        return [str(s) for s in self.rois.get("scanID_spatial", self.fs)]

    @property
    def domains(self) -> dict:
        return {str(k): v for k, v in self.rois.get("domain_ROInumber", {}).items()}


def read_pf(pf_dir, *, components: bool = False) -> PfFiles:
    """Read a ``PF`` folder the way the dataset does (restricted unpickler).

    ``components=True`` also loads ``denoised_trace_components.pkl``.
    """
    pf_dir = Path(pf_dir)
    if pf_dir.name != "PF" and (pf_dir / "PF").is_dir():
        pf_dir = pf_dir / "PF"
    traces = _restricted_pickle_load(pf_dir / TRACES_FILE)
    fs = _restricted_pickle_load(pf_dir / FS_FILE)
    rois = _restricted_pickle_load(pf_dir / ROIS_FILE)
    files = PfFiles(
        pf_dir=pf_dir,
        traces={str(k): {str(d): np.asarray(t) for d, t in v.items()} for k, v in traces.items()},
        fs={str(k): v for k, v in fs.items()},
        rois=dict(rois),
    )
    if (pf_dir / PEAKS_FILE).exists():
        peaks = _restricted_pickle_load(pf_dir / PEAKS_FILE)
        files.peaks = {str(k): {str(d): np.asarray(p, dtype=int) for d, p in v.items()} for k, v in peaks.items()}
    if (pf_dir / PARAMS_FILE).exists():
        files.params = dict(_restricted_pickle_load(pf_dir / PARAMS_FILE))
    if components and (pf_dir / COMPONENTS_FILE).exists():
        with (pf_dir / COMPONENTS_FILE).open("rb") as handle:
            raw = _ComponentsUnpickler(handle).load()
        files.components = {str(k): {str(d): dict(c) for d, c in v.items()} for k, v in raw.items()}
    if (pf_dir / PROVENANCE_FILE).exists():
        files.provenance = json.loads((pf_dir / PROVENANCE_FILE).read_text())
    return files
