"""Per-ROI fluorescence traces to a ``PF`` folder: stages 0-5 in one call.

The input is what any reader of the raw recording can provide: per scan, the
mean fluorescence of every ROI per frame and each ROI's pixel count
(:class:`ScanTraces`). :func:`process_scan` turns one scan into per-domain
z-scores and denoised traces; :func:`run_pipeline` writes the folder. The
archive's packaged pickles (``VI_<date>.pkl``) load through :func:`load_vi`.
"""

from __future__ import annotations

import importlib.metadata
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import hilbert

from .dataset import _restricted_pickle_load
from .denoiser import Denoiser, fir_lowpass
from .events import SpikeDetectConfig, detect_peaks
from .pf import DomainResult, PfWriter
from .preprocess import DfofConfig, domain_zscore

__all__ = [
    "ScanTraces",
    "ScanResult",
    "process_domain",
    "process_scan",
    "run_pipeline",
    "load_vi",
    "load_scan_rois",
    "run_from_vi",
]


@dataclass
class ScanTraces:
    """One scan's raw traces.

    Parameters
    ----------
    scan_id : str
        The scan's id as the pipeline keys it (the MUnit number).
    fs_hz : float
        The exact frame rate.
    traces : mapping of roi -> array (n_frame,)
        Mean fluorescence per ROI per frame, in the recording's counts.
    weights : mapping of roi -> float
        Pixel count of each ROI.
    comment : str
        The unit's comment, if any.
    """

    scan_id: str
    fs_hz: float
    traces: dict
    weights: dict
    comment: str = ""

    @property
    def n_frames(self) -> int:
        return int(len(next(iter(self.traces.values()))))

    @property
    def rois(self) -> list:
        return sorted(int(r) for r in self.traces)


@dataclass
class ScanResult:
    """One scan processed: the stage 0-1 traces and every domain's result."""

    scan_id: str
    names: list
    dfof_raw: np.ndarray
    z: np.ndarray
    domains: dict = field(default_factory=dict)


def process_domain(
    z,
    fs_hz: float,
    *,
    denoiser: Denoiser | None = None,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    keep_cwt: bool = False,
    cwt=None,
) -> DomainResult:
    """Stages 2-5 on one z-scored trace.

    Parameters
    ----------
    z : array (n_frame,)
    fs_hz : float
        The exact frame rate; the wavelet transform and FIR filters use it.
    denoiser : Denoiser, optional
        Defaults to :meth:`Denoiser.upstream`.
    spike_cfg : SpikeDetectConfig, optional
    detect : bool, default True
        Run the peak detector on the final trace.
    keep_cwt : bool, default False
        Keep the wavelet coefficients on the result (for ``cwts.h5``).
    cwt : tuple of (coefficients, frequencies), optional
        A saved transform of ``z`` to reuse (see :meth:`Denoiser.run`).
    """
    z = np.asarray(z, dtype=float).ravel()
    model = denoiser or Denoiser.upstream(fs_hz)
    model.run(z, cwt=cwt)
    timing = dict(model.timing_)
    lp1 = np.asarray(model.lp_dfof_, dtype=float)
    started = time.perf_counter()
    lp100 = fir_lowpass(z, model.fs, 100.0, model.fir_window_ms, odd_taps=model.fir_odd_taps)
    timing["baseline_100hz"] = time.perf_counter() - started
    result = DomainResult(
        rescaled_signal=np.asarray(model.rescaled_signal_),
        lp_fir1hz=lp1,
        lp_fir100hz=np.asarray(lp100, dtype=float),
        envelope_lp=np.abs(hilbert(lp1)),
        cwt=(model.coeff_, model.freqs_) if keep_cwt else None,
        timing=timing,
    )
    if detect:
        started = time.perf_counter()
        # the archive's notebooks ran the detector at floor(fs), the value in fs_scans.pkl
        result.peaks = detect_peaks(result.trace, float(np.floor(fs_hz)), spike_cfg)
        timing["peaks"] = time.perf_counter() - started
    model.coeff_ = None
    return result


def process_scan(
    scan: ScanTraces,
    domains: Mapping[str, Sequence[int]],
    *,
    dfof_cfg: DfofConfig | None = None,
    denoiser_factory: Callable[[float], Denoiser] | None = None,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    keep_cwt: bool = False,
    progress: Callable[[str, str], None] | None = None,
) -> ScanResult:
    """Stages 0-5 for every domain of one scan.

    ``denoiser_factory(fs_hz)`` builds the denoiser per domain (default
    :meth:`Denoiser.upstream`); ``progress(scan_id, domain)`` is called
    before each domain.
    """
    traces = domain_zscore(scan.traces, domains, scan.weights, dfof_cfg)
    result = ScanResult(
        scan_id=str(scan.scan_id), names=list(traces.names), dfof_raw=traces.dfof_raw, z=traces.z,
    )
    factory = denoiser_factory or Denoiser.upstream
    for row, name in enumerate(traces.names):
        if progress is not None:
            progress(result.scan_id, name)
        result.domains[name] = process_domain(
            traces.z[row], scan.fs_hz, denoiser=factory(scan.fs_hz),
            spike_cfg=spike_cfg, detect=detect, keep_cwt=keep_cwt,
        )
    return result


def run_pipeline(
    scans: Iterable[ScanTraces],
    pf_dir,
    *,
    domains: Mapping[str, Sequence[int]],
    first_env: Sequence[str] = (),
    dfof_cfg: DfofConfig | None = None,
    denoiser_factory: Callable[[float], Denoiser] | None = None,
    spike_cfg: SpikeDetectConfig | None = None,
    detect: bool = True,
    save_cwt: bool = False,
    provenance: dict | None = None,
    overwrite: bool = False,
    progress: Callable[[str, str], None] | None = None,
) -> dict:
    """Process every scan and write ``pf_dir``; returns ``{file: path}``.

    Parameters
    ----------
    scans : iterable of ScanTraces
        In ``scanID_spatial`` order.
    pf_dir : path
    domains : mapping of domain -> ROIs
    first_env : the first scan id of each environment
    dfof_cfg, denoiser_factory, spike_cfg, detect, save_cwt : see
        :func:`process_scan`
    provenance : dict, optional
        Recorded in ``pipeline.json`` (the source file, say).
    overwrite : bool
    progress : callable(scan_id, domain), optional
    """
    scans = list(scans)
    if not scans:
        raise ValueError("no scans")
    dfof_cfg = dfof_cfg or DfofConfig()
    spike_cfg = spike_cfg or SpikeDetectConfig()
    factory = denoiser_factory or Denoiser.upstream
    example = factory(scans[0].fs_hz)
    versions = {"python": platform.python_version()}
    for name in ("vnoiser", "numpy", "scipy", "scikit-learn", "PyWavelets", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = {
        "pipeline": "vnoiser.pipeline.run_pipeline",
        "versions": versions,
        "dfof": asdict(dfof_cfg),
        "denoiser": example.describe(),
        "events": asdict(spike_cfg) if detect else None,
        "comments": {s.scan_id: s.comment for s in scans if s.comment},
        "roi_weights": {s.scan_id: {int(k): float(v) for k, v in s.weights.items()} for s in scans},
    }
    info.update(provenance or {})
    writer = PfWriter(
        pf_dir,
        domains=domains,
        scan_ids=[s.scan_id for s in scans],
        first_env=first_env,
        roi_list={s.scan_id: s.rois for s in scans},
        spike_params=spike_cfg.to_param_pickle() if detect else None,
        save_cwt=save_cwt,
        freq_scales=example.freq_scales,
        provenance=info,
        overwrite=overwrite,
    )
    for scan in scans:
        result = process_scan(
            scan, domains, dfof_cfg=dfof_cfg, denoiser_factory=factory,
            spike_cfg=spike_cfg, detect=detect, keep_cwt=save_cwt, progress=progress,
        )
        writer.add_scan(scan.scan_id, scan.fs_hz, result.dfof_raw, result.z)
        for name, domain_result in result.domains.items():
            writer.add_domain(scan.scan_id, name, domain_result)
            domain_result.cwt = None
    return writer.finish()


def load_vi(path, *, channel: str = "green") -> tuple:
    """Read a packaged ``VI_<date>.pkl`` (``package_data*.py`` output).

    Returns ``(animal, experiment, {scan_id: ScanTraces})`` for the first
    animal and experiment in the file, as the notebooks took them. Each ROI's
    trace is the mean over its pixels of the raw channel; the weight is the
    pixel count.
    """
    payload = _restricted_pickle_load(Path(path))
    animal = str(next(iter(payload)))
    experiment = str(next(iter(payload[animal])))
    scans = {}
    for scan_key, entry in payload[animal][experiment].items():
        traces, weights, comment, fs = {}, {}, "", None
        for roi_key, roi in entry.items():
            if roi_key == "behavior":
                continue
            data = roi[channel]
            raw = np.asarray(data["raw"])
            frames = raw.reshape(raw.shape[0], -1).astype(float)
            traces[int(roi_key)] = frames.mean(axis=1)
            weights[int(roi_key)] = float(frames.shape[1])
            fs = float(data["framerate"])
            comment = str(data.get("comments", "") or "")
        if traces:
            scans[str(scan_key)] = ScanTraces(str(scan_key), fs, traces, weights, comment)
    return animal, experiment, scans


def load_scan_rois(path) -> dict:
    """Read ``scanIDs_ROIs.pkl``: ``domains``, ``scan_ids``, ``first_env``, ``roi_list``."""
    rois = _restricted_pickle_load(Path(path))
    return {
        "domains": {str(k): [int(v) for v in r] for k, r in rois["domain_ROInumber"].items()},
        "scan_ids": [str(s) for s in rois.get("scanID_spatial", [])],
        "first_env": [str(s) for s in rois.get("scanID_1st_env", [])],
        "roi_list": {str(k): [int(v) for v in r] for k, r in rois.get("roi_list", {}).items()},
    }


def run_from_vi(
    vi_path,
    pf_dir,
    *,
    domains: Mapping[str, Sequence[int]],
    scan_ids: Sequence[str] | None = None,
    first_env: Sequence[str] = (),
    **kwargs,
) -> dict:
    """:func:`run_pipeline` on the scans of a packaged ``VI_<date>.pkl``.

    ``scan_ids`` selects and orders the scans (default: every scan in the
    file, sorted numerically); other keyword arguments go to
    :func:`run_pipeline`.
    """
    animal, experiment, scans = load_vi(vi_path)
    if scan_ids is None:
        scan_ids = sorted(scans, key=lambda s: (len(s), s))
    missing = [s for s in scan_ids if str(s) not in scans]
    if missing:
        raise KeyError(f"scan(s) {missing} not in {vi_path}; it has {sorted(scans)}")
    provenance = {"source": {"vi": str(Path(vi_path)), "animal": animal, "experiment": experiment}}
    provenance.update(kwargs.pop("provenance", None) or {})
    return run_pipeline(
        [scans[str(s)] for s in scan_ids], pf_dir, domains=domains, first_env=first_env,
        provenance=provenance, **kwargs,
    )
