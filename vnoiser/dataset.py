"""Dataset helpers for local and BioHPC JEDI3sub recordings."""

from __future__ import annotations

import collections
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

try:
    from numpy._core import multiarray as _numpy_multiarray
except ImportError:  # NumPy < 2
    from numpy.core import multiarray as _numpy_multiarray


@dataclass
class RecordingSample:
    """A sampled window from one local JEDI3sub recording."""

    t: np.ndarray
    trace: np.ndarray
    fs_hz: float
    events_ap_indices: np.ndarray
    events_ap_times_s: np.ndarray
    path: Path
    metadata: dict


@dataclass(frozen=True)
class SpatialTraceRef:
    """One processed scan/domain trace inside a spatial JEDI experiment."""

    recording_id: str
    label: str
    animal: str
    experiment: str
    scan_id: str
    domain: str
    fs_hz: float
    pf_dir: Path
    trace_path: Path
    events_path: Optional[Path]


_SAFE_PICKLE_GLOBALS = {
    ("builtins", "dict"): dict,
    ("builtins", "str"): str,
    ("builtins", "list"): list,
    ("builtins", "range"): range,
    ("collections", "defaultdict"): collections.defaultdict,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
    ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _SAFE_PICKLE_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Unsupported pickle object {module}.{name}"
            ) from exc


def _restricted_pickle_load(path: Path):
    with path.open("rb") as handle:
        return _RestrictedUnpickler(handle).load()


def _pickle_cache_key(path: Path):
    stat = path.stat()
    return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _load_trace_pickle(path_string, _size, _mtime_ns):
    return _restricted_pickle_load(Path(path_string))


@lru_cache(maxsize=32)
def _load_small_pickle(path_string, _size, _mtime_ns):
    return _restricted_pickle_load(Path(path_string))


# the archive names the soma soma1 in scanIDs_ROIs.pkl and soma in the trace and peak files
DOMAIN_ALIASES = {"soma1": "soma", "soma": "soma1"}
_MISSING = object()


def _domain_entry(mapping, domain, default=_MISSING):
    """``mapping[domain]``, else the same under the domain's alias."""
    if domain in mapping:
        return mapping[domain]
    alias = DOMAIN_ALIASES.get(str(domain))
    if alias is not None and alias in mapping:
        return mapping[alias]
    if default is _MISSING:
        raise KeyError(domain)
    return default


@lru_cache(maxsize=128)
def _subdirectories(path_string):
    """List one directory level with a single SMB-friendly scandir call."""
    try:
        with os.scandir(path_string) as entries:
            paths = [Path(entry.path) for entry in entries if entry.is_dir()]
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return tuple()
    return tuple(sorted(paths, key=lambda path: path.name))


class JediSub3Dataset:
    """Loader for the local DS01 JEDI3sub MATLAB mini recordings.

    Parameters
    ----------
    root:
        Directory containing ``*_mini.mat`` files with a ``CAttached`` struct.
    pattern:
        Glob pattern used to discover recordings.
    """

    def __init__(self, root: str | Path = "DS01-JEDI-sub3", pattern: str = "*.mat"):
        self.root = Path(root)
        self.pattern = pattern
        self.recordings = tuple(sorted(self.root.glob(pattern)))
        if not self.recordings:
            raise FileNotFoundError(
                f"No JEDI3sub .mat recordings found under {self.root!s}"
            )

    def __len__(self) -> int:
        return len(self.recordings)

    def recording_options(self):
        return [(path.name, str(path)) for path in self.recordings]

    def sample(
        self,
        duration_s: Optional[float] = 10.0,
        seed: Optional[int] = None,
        require_events: bool = False,
    ) -> RecordingSample:
        """Randomly choose a recording and return a random time window."""
        rng = np.random.default_rng(seed)
        candidates = self.recordings
        if require_events:
            candidates = tuple(path for path in candidates if self._event_count(path) > 0)
            if not candidates:
                raise ValueError("No recordings with AP events were found")

        path = candidates[int(rng.integers(0, len(candidates)))]
        full = self.load(path)

        if duration_s is None:
            start = 0
            stop = len(full.trace)
        else:
            n_window = int(round(duration_s * full.fs_hz))
            if n_window <= 0:
                raise ValueError("duration_s must be positive")
            n_window = min(n_window, len(full.trace))
            max_start = len(full.trace) - n_window
            if require_events and len(full.events_ap_indices):
                event = int(full.events_ap_indices[int(rng.integers(0, len(full.events_ap_indices)))])
                start_min = max(0, event - n_window + 1)
                start_max = min(event, max_start)
                start = int(rng.integers(start_min, start_max + 1))
            else:
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            stop = start + n_window

        events = full.events_ap_indices
        window_events = events[(events >= start) & (events < stop)] - start
        t = full.t[start:stop] - full.t[start]
        trace = full.trace[start:stop]
        metadata = dict(full.metadata)
        metadata.update(
            {
                "window_start_s": float(full.t[start] - full.t[0]),
                "window_duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                "source_n_samples": int(len(full.trace)),
            }
        )
        return RecordingSample(
            t=t,
            trace=trace,
            fs_hz=full.fs_hz,
            events_ap_indices=window_events.astype(int, copy=False),
            events_ap_times_s=t[window_events] if len(window_events) else np.array([]),
            path=path,
            metadata=metadata,
        )

    def load(self, path: str | Path) -> RecordingSample:
        """Load a complete recording from disk."""
        path = Path(path)
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "CAttached" not in mat:
            raise KeyError(f"{path!s} does not contain a CAttached struct")

        attached = mat["CAttached"]
        trace = np.asarray(attached.fluo_mean, dtype=float).ravel()
        t = np.asarray(attached.fluo_time, dtype=float).ravel()
        if trace.size != t.size:
            raise ValueError(f"{path!s} has mismatched fluo_mean and fluo_time lengths")
        if trace.size < 2:
            raise ValueError(f"{path!s} has fewer than two samples")

        events = np.asarray(getattr(attached, "events_AP", []), dtype=int).ravel()
        events = events[(events >= 0) & (events < trace.size)]
        fs_hz = float(1.0 / np.median(np.diff(t)))
        return RecordingSample(
            t=t - t[0],
            trace=trace,
            fs_hz=fs_hz,
            events_ap_indices=events,
            events_ap_times_s=t[events] - t[0] if len(events) else np.array([]),
            path=path,
            metadata={
                "filename": path.name,
                "root": str(self.root),
                "fs_hz": fs_hz,
                "duration_s": float(t[-1] - t[0]),
                "n_samples": int(trace.size),
                "n_ap_events": int(events.size),
            },
        )

    @staticmethod
    def _event_count(path: Path) -> int:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        attached = mat["CAttached"]
        return int(np.asarray(getattr(attached, "events_AP", [])).size)


class SpatialJediDataset:
    """Progressive loader for processed spatial JEDI scan/domain traces.

    A shared ``Data`` or animal directory is traversed one level at a time.
    Scan/ROI metadata is read only after one experiment is selected. The large
    trace pickle is loaded only after one scan/domain is selected.
    """

    trace_filename = "denoised_trace_scans.pkl"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        scope = self._classify_root(self.root)
        if scope is None:
            raise FileNotFoundError(
                f"No spatial JEDI data structure found at {self.root!s}"
            )
        self.scope, self.fixed_experiment = scope
        self.recordings = tuple()
        self._by_id = {}
        if self.fixed_experiment is not None:
            self.select_experiment(self.fixed_experiment)

    def __len__(self):
        return len(self.recordings)

    @classmethod
    def clear_inventory_cache(cls):
        """Refresh directory listings after the user explicitly clicks Scan."""
        _subdirectories.cache_clear()
        _load_small_pickle.cache_clear()
        cls._index_experiment.cache_clear()

    @classmethod
    def _classify_root(cls, root: str | Path):
        root = Path(root).expanduser()
        if root.is_file():
            if root.name == cls.trace_filename:
                return "experiment", root.parent.parent
            return None
        if not root.is_dir():
            return None

        if root.name == "PF" and (root / cls.trace_filename).is_file():
            return "experiment", root.parent
        if (root / cls.trace_filename).is_file():
            return "experiment", root.parent
        if (root / "PF" / cls.trace_filename).is_file():
            return "experiment", root

        children = _subdirectories(str(root.resolve()))
        if any("_expt" in child.name for child in children):
            return "animal", None
        if any(child.name.startswith("stan") for child in children):
            return "data", None
        return None

    @classmethod
    def can_open(cls, root: str | Path):
        return cls._classify_root(root) is not None

    @property
    def requires_experiment_selection(self):
        return self.fixed_experiment is None

    def animal_options(self):
        """Return only the animal directories visible at the current level."""
        if self.scope == "data":
            animals = [
                path
                for path in _subdirectories(str(self.root.resolve()))
                if path.name.startswith("stan") and "_expt" not in path.name
            ]
        elif self.scope == "animal":
            animals = [self.root]
        else:
            animals = [self.fixed_experiment.parent]
        return [(path.name, str(path)) for path in animals]

    def experiment_options(self, animal=None):
        """List experiments without opening their PF metadata files."""
        if self.fixed_experiment is not None:
            experiments = [self.fixed_experiment]
        else:
            animal_path = Path(animal).expanduser() if animal else self.root
            experiments = [
                path
                for path in _subdirectories(str(animal_path.resolve()))
                if "_expt" in path.name
            ]
        return [(path.name, str(path)) for path in experiments]

    def select_experiment(self, experiment: str | Path):
        """Read metadata for one experiment and expose its scan/domain choices."""
        experiment = Path(experiment).expanduser()
        if experiment.name == "PF":
            experiment = experiment.parent
        trace_path = experiment / "PF" / self.trace_filename
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"{experiment!s} does not contain PF/{self.trace_filename}"
            )
        references = self._index_experiment(str(trace_path.resolve()))
        self.recordings = references
        self._by_id = {
            reference.recording_id: reference for reference in references
        }
        return self.recordings

    def recording_options(self):
        return [(reference.label, reference.recording_id) for reference in self.recordings]

    def load(
        self,
        recording: str | SpatialTraceRef,
        *,
        load_events: bool = True,
    ) -> RecordingSample:
        reference = self._resolve_reference(recording)
        traces = _load_trace_pickle(*_pickle_cache_key(reference.trace_path))
        traces_by_scan = {str(key): value for key, value in traces.items()}
        try:
            trace = np.asarray(
                _domain_entry(traces_by_scan[reference.scan_id], reference.domain),
                dtype=float,
            ).ravel()
        except KeyError as exc:
            raise KeyError(
                f"Missing trace for scan {reference.scan_id}, domain {reference.domain}"
            ) from exc
        if trace.size < 2:
            raise ValueError(f"{reference.label} has fewer than two samples")
        if not np.isfinite(trace).all():
            raise ValueError(f"{reference.label} contains NaN or infinite values")

        events = np.array([], dtype=int)
        if load_events and reference.events_path is not None:
            event_map = _load_small_pickle(*_pickle_cache_key(reference.events_path))
            events_by_scan = {str(key): value for key, value in event_map.items()}
            events = np.asarray(
                _domain_entry(events_by_scan.get(reference.scan_id, {}), reference.domain, []),
                dtype=int,
            ).ravel()
            events = np.unique(events[(events >= 0) & (events < trace.size)])

        t = np.arange(trace.size, dtype=float) / reference.fs_hz
        return RecordingSample(
            t=t,
            trace=trace,
            fs_hz=reference.fs_hz,
            events_ap_indices=events,
            events_ap_times_s=t[events] if len(events) else np.array([]),
            path=reference.trace_path,
            metadata={
                "recording_id": reference.recording_id,
                "label": reference.label,
                "animal": reference.animal,
                "experiment": reference.experiment,
                "scan_id": reference.scan_id,
                "domain": reference.domain,
                "fs_hz": reference.fs_hz,
                "duration_s": float(t[-1]),
                "n_samples": int(trace.size),
                "candidate_event_count": int(events.size),
                "candidate_event_source": (
                    str(reference.events_path)
                    if load_events and reference.events_path
                    else None
                ),
                "curation_dir": str(reference.pf_dir / ".curation"),
                "pre_denoised": True,
                "source_format": "spatial_jedi_pf",
            },
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def _index_experiment(trace_path_string):
        trace_path = Path(trace_path_string)
        pf_dir = trace_path.parent
        experiment_dir = pf_dir.parent
        animal_dir = experiment_dir.parent
        fs_path = pf_dir / "fs_scans.pkl"
        roi_path = pf_dir / "scanIDs_ROIs.pkl"
        if not fs_path.exists() or not roi_path.exists():
            return tuple()

        fs_by_scan = {
            str(key): value
            for key, value in _load_small_pickle(*_pickle_cache_key(fs_path)).items()
        }
        scan_metadata = _load_small_pickle(*_pickle_cache_key(roi_path))
        scan_ids = [str(value) for value in scan_metadata.get("scanID_spatial", [])]
        if not scan_ids:
            scan_ids = [str(value) for value in fs_by_scan]
        domain_map = scan_metadata.get("domain_ROInumber", {})
        domains = [str(key) for key in domain_map if str(key) != "All_domains"]
        events_path = pf_dir / "detected_events_peaks.pkl"
        if not events_path.exists():
            events_path = None

        references = []
        for scan_id in scan_ids:
            if scan_id not in fs_by_scan:
                continue
            for domain in domains:
                recording_id = (
                    f"{animal_dir.name}/{experiment_dir.name}/"
                    f"scan={scan_id}/domain={domain}"
                )
                references.append(
                    SpatialTraceRef(
                        recording_id=recording_id,
                        label=(
                            f"{animal_dir.name} / {experiment_dir.name} / "
                            f"scan {scan_id} / {domain}"
                        ),
                        animal=animal_dir.name,
                        experiment=experiment_dir.name,
                        scan_id=scan_id,
                        domain=domain,
                        fs_hz=float(fs_by_scan[scan_id]),
                        pf_dir=pf_dir,
                        trace_path=trace_path,
                        events_path=events_path,
                    )
                )
        return tuple(references)

    def _resolve_reference(self, recording):
        if isinstance(recording, SpatialTraceRef):
            return recording
        try:
            return self._by_id[str(recording)]
        except KeyError as exc:
            raise KeyError(f"Unknown spatial JEDI recording: {recording}") from exc


def open_recording_dataset(data_path: str | Path, pattern: str = "*.mat"):
    """Open the supported dataset represented by ``data_path``."""
    data_path = Path(data_path).expanduser()
    if SpatialJediDataset.can_open(data_path):
        return SpatialJediDataset(data_path)

    if data_path.exists() and data_path.is_file():
        root = data_path.parent
        pattern = data_path.name
    else:
        root = data_path
    return JediSub3Dataset(root, pattern=pattern)
