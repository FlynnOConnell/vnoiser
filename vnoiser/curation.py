"""Interactive event curation dashboard for vnoiser notebooks.

The dashboard is intentionally lazy: constructing or displaying it only lists
one directory level. Experiment metadata is opened after an experiment is
selected, and the trace or denoiser pipeline is loaded only after the user
clicks Load. Cached pipeline results live next to the data in
``.curation/cache``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.decomposition import PCA

from .dataset import RecordingSample, SpatialJediDataset, open_recording_dataset
from .denoiser import ClusteringConfig, Denoiser, cwtReducerConfig, thresConfig


LABEL_COLORS = {
    "yes": "#1b998b",
    "no": "#d7263d",
    "auto_yes": "#91d7c9",
    "auto_no": "#f3a0aa",
    "unlabeled": "#525866",
}

AUTO_TEMPLATE_THRESHOLD = 0.8
PIPELINE_CACHE_VERSION = 1
PANEL_WIDTH_PX = 520
PANEL_GAP_PX = 16
DASHBOARD_WIDTH_PX = PANEL_WIDTH_PX * 3 + PANEL_GAP_PX * 2
THRESHOLD_PANEL_WIDTH_PX = 150
TIMELINE_PLOT_WIDTH_PX = (
    DASHBOARD_WIDTH_PX - THRESHOLD_PANEL_WIDTH_PX - PANEL_GAP_PX
)
TIMELINE_HEIGHT_PX = 260
PANEL_HEIGHT_PX = 285
CONTROL_PANEL_HEIGHT_PX = 255
DECISION_BUTTON_WIDTH_PX = 150
DECISION_BUTTON_HEIGHT_PX = 56
DECISION_BUTTON_GAP_PX = 12
DECISION_ROW_HEIGHT_PX = 68
DECISION_BUTTON_STYLE_HTML = f"""
<style>
.vnoiser-curation-dashboard,
.vnoiser-curation-dashboard * {{
  box-sizing: border-box;
}}
.vnoiser-curation-dashboard .vnoiser-control-panel {{
  box-sizing: border-box !important;
}}
.vnoiser-curation-dashboard .vnoiser-decision-row {{
  align-items: center !important;
  display: flex !important;
  flex-flow: row nowrap !important;
  gap: {DECISION_BUTTON_GAP_PX}px !important;
  height: {DECISION_ROW_HEIGHT_PX}px !important;
  min-height: {DECISION_ROW_HEIGHT_PX}px !important;
  overflow: visible !important;
  visibility: visible !important;
  width: {DECISION_BUTTON_WIDTH_PX * 3 + DECISION_BUTTON_GAP_PX * 2}px !important;
}}
.vnoiser-curation-dashboard .vnoiser-decision-action,
.vnoiser-curation-dashboard .vnoiser-decision-action button {{
  align-items: center !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  flex: 0 0 {DECISION_BUTTON_WIDTH_PX}px !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  height: {DECISION_BUTTON_HEIGHT_PX}px !important;
  justify-content: center !important;
  line-height: 24px !important;
  max-height: {DECISION_BUTTON_HEIGHT_PX}px !important;
  max-width: {DECISION_BUTTON_WIDTH_PX}px !important;
  min-height: {DECISION_BUTTON_HEIGHT_PX}px !important;
  min-width: {DECISION_BUTTON_WIDTH_PX}px !important;
  overflow: visible !important;
  padding: 0 18px !important;
  visibility: visible !important;
  width: {DECISION_BUTTON_WIDTH_PX}px !important;
}}
.vnoiser-curation-dashboard .vnoiser-decision-yes,
.vnoiser-curation-dashboard .vnoiser-decision-yes button {{
  background: #1b998b !important;
  border-color: #1b998b !important;
  color: #ffffff !important;
}}
.vnoiser-curation-dashboard .vnoiser-decision-no,
.vnoiser-curation-dashboard .vnoiser-decision-no button {{
  background: #d7263d !important;
  border-color: #d7263d !important;
  color: #ffffff !important;
}}
.vnoiser-curation-dashboard .vnoiser-decision-clear,
.vnoiser-curation-dashboard .vnoiser-decision-clear button {{
  background: #e8e8e8 !important;
  border-color: #cfcfcf !important;
  color: #222222 !important;
}}
.vnoiser-curation-dashboard .vnoiser-wrap-path {{
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}}
</style>
"""


@dataclass
class CandidateSet:
    indices: np.ndarray
    aligned_indices: np.ndarray
    times_s: np.ndarray
    amplitudes: np.ndarray
    short_snippets: np.ndarray
    long_snippets: np.ndarray
    long_time_ms: np.ndarray
    template_time_ms: np.ndarray
    pca_scores: np.ndarray
    pca_explained_variance: np.ndarray


def robust_zscore(values, axis=None):
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(values - median), axis=axis, keepdims=True)
    std = np.std(values, axis=axis, keepdims=True)
    scale = np.where(mad > 0, mad, np.where(std > 0, std, 1.0))
    return (values - median) / scale


def cosine_similarity_rows(rows, template):
    rows = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0)
    template = np.nan_to_num(np.asarray(template, dtype=float), nan=0.0)
    if rows.size == 0 or template.size == 0:
        return np.zeros(rows.shape[0], dtype=float)
    centered_rows = rows - np.mean(rows, axis=1, keepdims=True)
    centered_template = template - np.mean(template)
    denom = np.linalg.norm(centered_rows, axis=1) * np.linalg.norm(centered_template)
    return np.divide(
        centered_rows @ centered_template,
        denom,
        out=np.zeros(rows.shape[0], dtype=float),
        where=denom > 0,
    )


def downsample_xy(x, y, max_points=3500):
    step = max(1, int(np.ceil(len(x) / max_points)))
    return x[::step], y[::step]


def default_data_path():
    candidates = [Path("DS01-JEDI-sub3"), Path("../DS01-JEDI-sub3")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd()


def curation_dir_for(data_path):
    path = Path(data_path).expanduser()
    if path.exists() and path.is_file():
        return path.parent / ".curation"
    if path.suffix:
        return path.parent / ".curation"
    return path / ".curation"


def consolidated_pca_event_calls(
    masked_cluster_traces,
    reference_trace,
    fs_hz,
    threshold_z=3.0,
    prominence_z=1.0,
    min_distance_ms=6.0,
    smooth_ms=1.5,
):
    features = np.asarray(masked_cluster_traces, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        return np.array([], dtype=int), np.zeros_like(reference_trace), threshold_z

    feature_z = robust_zscore(features, axis=1)
    score = PCA(n_components=1).fit_transform(feature_z.T).ravel()
    corr = np.corrcoef(score, reference_trace)[0, 1] if np.std(score) > 0 else 0
    if np.isfinite(corr) and corr < 0:
        score *= -1

    sigma = max(1, int(round((smooth_ms / 1000.0) * fs_hz)))
    score_z = robust_zscore(gaussian_filter1d(score, sigma=sigma)).ravel()
    distance = max(1, int(round((min_distance_ms / 1000.0) * fs_hz)))
    peaks, _ = find_peaks(
        score_z,
        height=threshold_z,
        prominence=prominence_z,
        distance=distance,
    )
    return peaks.astype(int), score_z, threshold_z


def lowpass_trace(trace, fs_hz, cutoff_hz=40.0, order=4):
    """Return a zero-phase low-pass view used by slow-event curation."""
    trace = np.asarray(trace, dtype=float).ravel()
    if trace.size < 2:
        return trace.copy()
    nyquist_hz = 0.5 * float(fs_hz)
    cutoff_hz = min(float(cutoff_hz), nyquist_hz * 0.95)
    if cutoff_hz <= 0:
        raise ValueError("slow_cutoff_hz must be positive")
    sos = butter(int(order), cutoff_hz / nyquist_hz, btype="low", output="sos")
    padlen = min(trace.size - 1, 3 * (2 * len(sos) + 1))
    if padlen < 1:
        return trace.copy()
    return sosfiltfilt(sos, trace, padlen=padlen)


def threshold_event_calls(
    trace,
    fs_hz,
    threshold,
    min_distance_ms,
):
    """Detect positive local maxima above a user-controlled trace threshold."""
    trace = np.asarray(trace, dtype=float).ravel()
    distance = max(1, int(round((min_distance_ms / 1000.0) * fs_hz)))
    peaks, _ = find_peaks(
        trace,
        height=float(threshold),
        distance=distance,
    )
    return peaks.astype(int)


def threshold_slider_scale(
    trace,
    fs_hz,
    min_distance_ms,
    selected_threshold=None,
):
    """Return a useful data-derived threshold range and initialization."""
    trace = np.asarray(trace, dtype=float).ravel()
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return 0.0, 1.0, 0.5, 0.01

    center = float(np.median(finite))
    sigma = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = max(abs(center) * 0.01, 1e-3)

    distance = max(1, int(round((min_distance_ms / 1000.0) * fs_hz)))
    peak_indices, _ = find_peaks(trace, distance=distance)
    peak_values = trace[peak_indices]
    peak_values = peak_values[np.isfinite(peak_values) & (peak_values >= center)]

    automatic = center + 3.0 * sigma
    if peak_values.size:
        upper_quantile = 100.0 if peak_values.size < 20 else 99.0
        robust_upper = float(np.percentile(peak_values, upper_quantile))
        data_upper = min(float(np.max(peak_values)), robust_upper + 5.0 * sigma)
    else:
        data_upper = float(np.max(finite))

    lower = center
    upper = max(data_upper, automatic + sigma, lower + sigma)
    upper += 0.02 * max(upper - lower, sigma)

    if selected_threshold is None or not np.isfinite(selected_threshold):
        selected = float(np.clip(automatic, lower, upper))
    else:
        selected = float(selected_threshold)
        lower = min(lower, selected)
        upper = max(upper, selected)

    step = max((upper - lower) / 100.0, np.finfo(float).eps)
    return lower, upper, selected, step


def candidate_pca_embedding(long_snippets, long_time_ms, window_ms=400.0):
    """Project centered candidate windows into a two-dimensional PCA space."""
    snippets = np.asarray(long_snippets, dtype=float)
    time_ms = np.asarray(long_time_ms, dtype=float)
    n_events = snippets.shape[0] if snippets.ndim == 2 else 0
    empty_scores = np.zeros((n_events, 2), dtype=float)
    empty_variance = np.zeros(2, dtype=float)
    if n_events == 0 or time_ms.size == 0:
        return empty_scores, empty_variance

    half_window = float(window_ms) / 2.0
    window = (time_ms >= -half_window) & (time_ms <= half_window)
    if not np.any(window):
        return empty_scores, empty_variance

    rows = snippets[:, window].copy()
    if rows.shape[1] > 512:
        step = int(np.ceil(rows.shape[1] / 512))
        rows = rows[:, ::step]
    sample_positions = np.arange(rows.shape[1])
    for row in rows:
        valid = np.isfinite(row)
        if np.any(valid):
            row[~valid] = np.interp(
                sample_positions[~valid],
                sample_positions[valid],
                row[valid],
            )
        else:
            row[:] = 0.0
    rows -= np.mean(rows, axis=1, keepdims=True)
    if n_events < 2 or np.allclose(rows, rows[0]):
        return empty_scores, empty_variance

    n_components = min(2, rows.shape[0], rows.shape[1])
    solver = "randomized" if min(rows.shape) > n_components else "full"
    model = PCA(
        n_components=n_components,
        svd_solver=solver,
        random_state=0 if solver == "randomized" else None,
    )
    transformed = model.fit_transform(rows)
    scores = empty_scores
    scores[:, :n_components] = transformed
    explained = empty_variance
    explained[:n_components] = np.nan_to_num(model.explained_variance_ratio_)
    return scores, explained


class EventCurationDashboard:
    """Widget dashboard for manual event curation.

    Parameters
    ----------
    data_path:
        A raw ``.mat`` source or a spatial JEDI data/PF directory. Labels and
        cache are written under the selected data source's ``.curation`` folder.
    mode:
        ``"manual"`` starts empty, ``"fast"`` seeds from high-amplitude fast
        candidates detected on the denoised trace, and ``"slow"`` seeds from
        candidates detected on a zero-phase less-than-40-Hz view.
    auto_load:
        If true, process the selected recording immediately. If false, only
        build the UI; the pipeline runs after the user clicks Load.
    """

    def __init__(
        self,
        data_path=None,
        mode="fast",
        label_path=None,
        duration_s=5.0,
        seed=21,
        dataset_root=None,
        file_pattern="*.mat",
        high_amplitude_quantile=0.75,
        fast_threshold=None,
        fast_min_distance_ms=6.0,
        slow_cutoff_hz=40.0,
        slow_threshold=None,
        slow_min_distance_ms=100.0,
        auto_load=False,
        enable_pipeline_cache=True,
    ):
        if mode not in {"manual", "fast", "slow"}:
            raise ValueError("mode must be 'manual', 'fast', or 'slow'")

        if data_path is None:
            data_path = dataset_root if dataset_root is not None else default_data_path()

        self.mode = mode
        self.duration_s = None if duration_s is None else float(duration_s)
        self.seed = int(seed)
        self.file_pattern = file_pattern
        self.high_amplitude_quantile = float(high_amplitude_quantile)
        self.fast_threshold = (
            None if fast_threshold is None else float(fast_threshold)
        )
        self.fast_min_distance_ms = float(fast_min_distance_ms)
        self.slow_cutoff_hz = float(slow_cutoff_hz)
        self.slow_threshold = (
            None if slow_threshold is None else float(slow_threshold)
        )
        self.slow_min_distance_ms = float(slow_min_distance_ms)
        self.candidate_threshold = (
            self.slow_threshold if mode == "slow" else self.fast_threshold
        )
        self.enable_pipeline_cache = bool(enable_pipeline_cache)
        self.explicit_label_path = Path(label_path).expanduser() if label_path else None

        self.dataset = None
        self.recordings = tuple()
        self.data_path = Path(data_path).expanduser()
        self.curation_dir = curation_dir_for(self.data_path)
        self.label_path = self._default_label_path()
        self.saved_events = {}
        self.saved_candidate_detection = {}
        self.labels = {}
        self.pipeline_cache_status = "not loaded"

        self.recording = None
        self.raw_trace = np.array([], dtype=float)
        self.denoiser_input = np.array([], dtype=float)
        self.denoised = np.array([], dtype=float)
        self.analysis_trace = np.array([], dtype=float)
        self.raw_cluster_starts = np.array([], dtype=int)
        self.pca_event_indices = np.array([], dtype=int)
        self.pca_event_score_z = np.array([], dtype=float)
        self.pca_threshold_z = np.nan
        self.threshold_event_indices = np.array([], dtype=int)
        self.retained_event_indices = np.array([], dtype=int)
        self.candidate_event_indices = np.array([], dtype=int)
        self.candidates = self._empty_candidate_set(fs_hz=1.0)
        self.event_keys = []
        self.current = 0
        self.seed_indices = np.array([], dtype=int)
        self.initial_template_source = np.array([], dtype=int)
        self.initial_template_scores = np.array([], dtype=float)
        self.template_source = np.array([], dtype=int)
        self.template = None
        self.template_scores = np.array([], dtype=float)
        self.second_pass = np.array([], dtype=float)
        self.visible_indices = np.array([], dtype=int)
        self.timeline_trace_indices = np.array([], dtype=int)
        self._updating_dataset_controls = False
        self._updating_threshold_slider = False

        self._configure_data_path(self.data_path)
        self._build_widgets()
        self._set_loaded_controls(False)
        self._draw_empty_all(self._initial_message())

        if auto_load and self.recordings:
            self._load_selected_recording(None)

    def display(self):
        display(self.ui)

    def _default_label_path(self):
        if self.explicit_label_path is not None:
            return self.explicit_label_path
        return self.curation_dir / f"{self.mode}_template_curation.json"

    def _initial_message(self):
        if (
            isinstance(self.dataset, SpatialJediDataset)
            and self.dataset.requires_experiment_selection
            and not self.recordings
        ):
            return "Select an animal and experiment. Metadata loads one experiment at a time."
        if not self.recordings:
            return "Enter a supported data file or folder, then click Scan."
        return "Select one recording, then click Load. No pipeline has run yet."

    def _configure_data_path(self, data_path):
        self.data_path = Path(data_path).expanduser()
        self.curation_dir = curation_dir_for(self.data_path)
        self.label_path = self._default_label_path()
        self.saved_events, self.labels = self._load_label_data()

        try:
            self.dataset = open_recording_dataset(
                self.data_path,
                pattern=self.file_pattern,
            )
        except FileNotFoundError:
            self.dataset = None
        self.recordings = self.dataset.recordings if self.dataset is not None else tuple()

    def _recording_options(self):
        if not self.recordings:
            if (
                isinstance(self.dataset, SpatialJediDataset)
                and self.dataset.requires_experiment_selection
            ):
                return [("Select an experiment first", "")]
            return [("No supported recordings found", "")]
        return self.dataset.recording_options()

    def _load_label_data(self):
        load_path = self._label_load_path()
        if load_path is None:
            self.saved_candidate_detection = {}
            return {}, {}
        with load_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.saved_candidate_detection = payload.get("candidate_detection", {})
        if self.recording is not None:
            recording_id = self._recording_label_id(self.recording.path)
            saved_threshold = self.saved_candidate_detection.get(
                "thresholds",
                {},
            ).get(recording_id)
            try:
                saved_threshold = float(saved_threshold)
            except (TypeError, ValueError):
                saved_threshold = np.nan
            if np.isfinite(saved_threshold):
                self.candidate_threshold = saved_threshold
        events = payload.get("events", {})
        labels = {
            key: value.get("label", "unlabeled")
            for key, value in events.items()
            if value.get("label", "unlabeled") != "unlabeled"
        }
        return events, labels

    def _label_load_path(self):
        if self.label_path.exists():
            return self.label_path
        return None

    def _save_labels(self):
        if self.recording is None:
            return

        events = dict(self.saved_events)
        threshold_indices = set(self.threshold_event_indices.tolist())
        seed_indices = set(self.seed_indices.tolist())
        for i, key in enumerate(self.event_keys):
            label = self.labels.get(key, "unlabeled")
            if label == "unlabeled":
                events.pop(key, None)
                continue

            source_offset = int(self.recording.metadata.get("window_start_index", 0))
            source_event_index = source_offset + int(self.candidates.indices[i])
            source_aligned_index = source_offset + int(self.candidates.aligned_indices[i])
            source_event_time = float(
                self.recording.metadata.get("window_start_s", 0.0)
                + self.candidates.times_s[i]
            )
            score = self.template_scores[i] if len(self.template_scores) else np.nan
            initial_score = self._initial_template_score_for_index(i)
            initial_call = self._initial_auto_call_for_index(i)
            candidate_source = (
                "threshold"
                if int(self.candidates.indices[i]) in threshold_indices
                else "retained_manual"
            )
            events[key] = {
                "label": label,
                "manual_label": label,
                "curation_state": "manual",
                "mode": self.mode,
                "recording": self._recording_label_id(self.recording.path),
                "filename": self.recording.path.name,
                "source_event_index": int(source_event_index),
                "source_aligned_index": int(source_aligned_index),
                "source_event_time_s": source_event_time,
                "window_event_index": int(self.candidates.indices[i]),
                "window_aligned_index": int(self.candidates.aligned_indices[i]),
                "window_event_time_s": float(self.candidates.times_s[i]),
                "window_start_s": float(self.recording.metadata.get("window_start_s", 0.0)),
                "window_start_index": source_offset,
                "amplitude": float(self.candidates.amplitudes[i]),
                "candidate_source": candidate_source,
                "candidate_threshold": float(self.candidate_threshold),
                "template_cosine": None if not np.isfinite(score) else float(score),
                "initial_template_cosine": (
                    None if not np.isfinite(initial_score) else float(initial_score)
                ),
                "initial_auto_call": initial_call,
                "auto_template_threshold": (
                    AUTO_TEMPLATE_THRESHOLD if self.mode in {"fast", "slow"} else None
                ),
                "was_seed_template_source": bool(i in seed_indices),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }

        self.saved_events = events
        recording_id = self._recording_label_id(self.recording.path)
        saved_thresholds = dict(
            self.saved_candidate_detection.get("thresholds", {})
        )
        saved_thresholds[recording_id] = float(self.candidate_threshold)
        candidate_detection = {
            "source": (
                "lowpass_trace" if self.mode == "slow" else "denoised_trace"
            ),
            "thresholds": saved_thresholds,
            "slow_cutoff_hz": (
                float(self.slow_cutoff_hz) if self.mode == "slow" else None
            ),
        }
        self.saved_candidate_detection = candidate_detection
        payload = {
            "version": 4,
            "mode": self.mode,
            "data_path": str(self.data_path),
            "candidate_detection": candidate_detection,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "events": events,
        }
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _recording_label_id(self, path):
        if self.recording is not None:
            recording_id = self.recording.metadata.get("recording_id")
            if recording_id:
                return str(recording_id)
        path = Path(path)
        try:
            base = self.data_path if self.data_path.is_dir() else self.data_path.parent
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            return path.name

    def _activate_recording_storage(self, recording):
        curation_dir = recording.metadata.get("curation_dir")
        if curation_dir and self.explicit_label_path is None:
            self.curation_dir = Path(curation_dir)
            self.label_path = self._default_label_path()
        self.recording = recording
        self.candidate_threshold = (
            self.slow_threshold if self.mode == "slow" else self.fast_threshold
        )
        self.saved_events, self.labels = self._load_label_data()

    def _window_from_recording(self, full):
        if self.duration_s is None:
            start = 0
            stop = len(full.trace)
        else:
            if self.duration_s <= 0:
                raise ValueError("duration_s must be positive")
            n_window = min(int(round(self.duration_s * full.fs_hz)), len(full.trace))
            if n_window >= len(full.trace):
                start = 0
            elif len(full.events_ap_indices):
                event = int(full.events_ap_indices[len(full.events_ap_indices) // 2])
                start = min(max(0, event - n_window // 2), len(full.trace) - n_window)
            else:
                start = 0
            stop = start + n_window

        events = full.events_ap_indices
        window_events = events[(events >= start) & (events < stop)] - start
        t = full.t[start:stop] - full.t[start]
        metadata = dict(full.metadata)
        metadata.update(
            {
                "window_start_s": float(full.t[start] - full.t[0]),
                "window_start_index": int(start),
                "window_duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                "source_n_samples": int(len(full.trace)),
            }
        )
        return RecordingSample(
            t=t,
            trace=full.trace[start:stop],
            fs_hz=full.fs_hz,
            events_ap_indices=window_events.astype(int, copy=False),
            events_ap_times_s=t[window_events] if len(window_events) else np.array([]),
            path=full.path,
            metadata=metadata,
        )

    def _pipeline_cache_path(self, recording):
        if not self.enable_pipeline_cache:
            return None
        stat = Path(recording.path).stat()
        payload = {
            "version": PIPELINE_CACHE_VERSION,
            "path": str(Path(recording.path).resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "fs_hz": float(recording.fs_hz),
            "n_samples": int(len(recording.trace)),
            "window_start_index": int(recording.metadata.get("window_start_index", 0)),
            "duration_s": self.duration_s,
            "mode_independent": True,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        safe_stem = Path(recording.path).stem.replace(" ", "_")
        return self.curation_dir / "cache" / f"{safe_stem}-{digest}.npz"

    def _run_or_load_pipeline(self, recording):
        if recording.metadata.get("pre_denoised"):
            self.recording = recording
            self.raw_trace = recording.trace.astype(float)
            self.denoiser_input = self.raw_trace.copy()
            self.denoised = self.raw_trace.copy()
            self.raw_cluster_starts = np.array([], dtype=int)
            self.pca_event_indices = np.array([], dtype=int)
            self.pca_event_score_z = np.zeros_like(self.raw_trace)
            self.pca_threshold_z = np.nan
            self.model = None
            self.pipeline_cache_status = "loaded processed denoised trace"
            self._finalize_pipeline_state()
            return

        cache_path = self._pipeline_cache_path(recording)
        if cache_path is not None and cache_path.exists():
            self._load_pipeline_cache(recording, cache_path)
            self.pipeline_cache_status = f"loaded cache: {cache_path.name}"
            return

        self._run_pipeline(recording)
        self.pipeline_cache_status = "computed pipeline"
        if cache_path is not None:
            self._save_pipeline_cache(cache_path)

    def _run_pipeline(self, recording):
        self.recording = recording
        self.raw_trace = recording.trace.astype(float)
        raw_std = np.std(self.raw_trace)
        if raw_std == 0:
            raise ValueError("recording trace has zero variance")
        self.denoiser_input = (self.raw_trace - np.median(self.raw_trace)) / raw_std
        self.model = Denoiser(
            fs=recording.fs_hz,
            freq_scales=np.logspace(np.log10(2), np.log10(1800), num=64),
            lp_cutoff=20.0,
            fir_window_ms=80.0,
            cfg_clust=ClusteringConfig(
                n_components=16,
                n_comp_clu=8,
                n_clusters=5,
                n_subclusters=8,
            ),
            cfg_reducer=cwtReducerConfig(slow_upthres=1.5, fast_upthres=2.0),
            cfg_thres=thresConfig(thres_type="soft"),
        )
        self.denoised, self.raw_cluster_starts = self.model.run(self.denoiser_input)
        self.denoised = np.real(self.denoised)
        self.pca_event_indices, self.pca_event_score_z, self.pca_threshold_z = (
            consolidated_pca_event_calls(
                self.model.threshold_result_["all_scaled"],
                self.denoiser_input,
                recording.fs_hz,
            )
        )
        self._finalize_pipeline_state()

    def _finalize_pipeline_state(self):
        self.analysis_trace = (
            lowpass_trace(
                self.denoised,
                self.recording.fs_hz,
                cutoff_hz=self.slow_cutoff_hz,
            )
            if self.mode == "slow"
            else self.denoised.copy()
        )
        self._configure_threshold_slider()
        self._rebuild_candidates()

    def _save_pipeline_cache(self, cache_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            denoiser_input=self.denoiser_input,
            denoised=self.denoised,
            raw_cluster_starts=np.asarray(self.raw_cluster_starts, dtype=int),
            pca_event_indices=self.pca_event_indices,
            pca_event_score_z=self.pca_event_score_z,
            pca_threshold_z=np.asarray([self.pca_threshold_z], dtype=float),
        )

    def _load_pipeline_cache(self, recording, cache_path):
        self.recording = recording
        self.raw_trace = recording.trace.astype(float)
        with np.load(cache_path, allow_pickle=False) as data:
            self.denoiser_input = data["denoiser_input"]
            self.denoised = data["denoised"]
            self.raw_cluster_starts = data["raw_cluster_starts"]
            self.pca_event_indices = data["pca_event_indices"]
            self.pca_event_score_z = data["pca_event_score_z"]
            self.pca_threshold_z = float(data["pca_threshold_z"][0])
        self.model = None
        self._finalize_pipeline_state()

    def _event_key(self, i):
        source_offset = int(self.recording.metadata.get("window_start_index", 0))
        source_index = source_offset + int(self.candidates.indices[i])
        recording_id = self._recording_label_id(self.recording.path)
        return f"{recording_id}|sample={source_index}"

    def _retained_manual_event_indices(self):
        """Return labeled events for this recording, even below the threshold."""
        if self.recording is None:
            return np.array([], dtype=int)

        recording_id = self._recording_label_id(self.recording.path)
        source_offset = int(self.recording.metadata.get("window_start_index", 0))
        retained = []
        for key, event in self.saved_events.items():
            if event.get("label") not in {"yes", "no"}:
                continue
            if (
                event.get("recording") != recording_id
                and not key.startswith(f"{recording_id}|sample=")
            ):
                continue
            source_index = event.get("source_event_index")
            if source_index is None:
                try:
                    source_index = int(key.rsplit("|sample=", 1)[1])
                except (IndexError, ValueError):
                    continue
            window_index = int(source_index) - source_offset
            if 0 <= window_index < len(self.analysis_trace):
                retained.append(window_index)
        return np.unique(np.asarray(retained, dtype=int))

    def _candidate_min_distance_ms(self):
        return (
            self.slow_min_distance_ms
            if self.mode == "slow"
            else self.fast_min_distance_ms
        )

    def _rebuild_candidates(self, preserve_key=None):
        if preserve_key is None and self.event_keys:
            preserve_key = self.event_keys[self.current]

        self.threshold_event_indices = threshold_event_calls(
            self.analysis_trace,
            self.recording.fs_hz,
            threshold=self.candidate_threshold,
            min_distance_ms=self._candidate_min_distance_ms(),
        )
        self.retained_event_indices = self._retained_manual_event_indices()
        self.candidate_event_indices = np.union1d(
            self.threshold_event_indices,
            self.retained_event_indices,
        ).astype(int, copy=False)
        self.candidates = self._build_candidates(self.candidate_event_indices)
        self.event_keys = [
            self._event_key(i) for i in range(len(self.candidates.indices))
        ]

        if preserve_key in self.event_keys:
            self.current = self.event_keys.index(preserve_key)
        else:
            self.current = 0
        self.seed_indices = self._initial_seed_indices()
        self._compute_initial_template_state()
        self.visible_indices = np.arange(len(self.candidates.indices), dtype=int)
        if hasattr(self, "view_filter"):
            self._refresh_visible(reselect=True, redraw=False)
        if hasattr(self, "event_slider"):
            self.event_slider.unobserve(self._slider_changed, names="value")
            try:
                self.event_slider.max = max(0, len(self.candidates.indices) - 1)
                self.event_slider.value = self.current
            finally:
                self.event_slider.observe(self._slider_changed, names="value")
    def _configure_threshold_slider(self):
        lower, upper, selected, step = threshold_slider_scale(
            self.analysis_trace,
            self.recording.fs_hz,
            self._candidate_min_distance_ms(),
            selected_threshold=self.candidate_threshold,
        )
        self.candidate_threshold = selected

        self._updating_threshold_slider = True
        try:
            self.threshold_slider.max = upper
            self.threshold_slider.min = lower
            self.threshold_slider.step = step
            self.threshold_slider.readout_format = ".2f"
            self.threshold_slider.value = self.candidate_threshold
        finally:
            self._updating_threshold_slider = False

    def _threshold_changed(self, change):
        if self._updating_threshold_slider or self.recording is None:
            return
        preserve_key = self.event_keys[self.current] if self.event_keys else None
        self.candidate_threshold = float(change["new"])
        self._rebuild_candidates(preserve_key=preserve_key)
        self._set_loaded_controls(True)
        self._refresh_all()
        self._save_labels()
        self.status.value = (
            f"<b>Status:</b> threshold updated to {self.candidate_threshold:.2f}; "
            f"{len(self.threshold_event_indices)} detected and "
            f"{len(self.retained_event_indices)} manually retained."
        )

    def _empty_candidate_set(self, fs_hz):
        short_pre, short_post, long_pre, long_post, _ = (
            self._candidate_window_samples(fs_hz)
        )
        long_time_ms = (np.arange(long_pre + long_post + 1) - long_pre) / fs_hz * 1000.0
        template_time_ms = (
            (np.arange(short_pre + short_post + 1) - short_pre) / fs_hz * 1000.0
        )
        return CandidateSet(
            indices=np.array([], dtype=int),
            aligned_indices=np.array([], dtype=int),
            times_s=np.array([], dtype=float),
            amplitudes=np.array([], dtype=float),
            short_snippets=np.empty((0, len(template_time_ms)), dtype=float),
            long_snippets=np.empty((0, len(long_time_ms)), dtype=float),
            long_time_ms=long_time_ms,
            template_time_ms=template_time_ms,
            pca_scores=np.empty((0, 2), dtype=float),
            pca_explained_variance=np.zeros(2, dtype=float),
        )

    def _candidate_window_samples(self, fs_hz):
        if self.mode == "slow":
            short_pre = max(1, int(round(0.500 * fs_hz)))
            short_post = max(1, int(round(0.500 * fs_hz)))
            align = max(1, int(round(0.025 * fs_hz)))
        else:
            short_pre = max(1, int(round(0.004 * fs_hz)))
            short_post = max(1, int(round(0.010 * fs_hz)))
            align = max(1, int(round(0.003 * fs_hz)))
        long_pre = max(1, int(round(0.500 * fs_hz)))
        long_post = max(1, int(round(0.500 * fs_hz)))
        return short_pre, short_post, long_pre, long_post, align

    def _build_candidates(self, event_indices):
        fs_hz = self.recording.fs_hz
        short_pre, short_post, long_pre, long_post, align = (
            self._candidate_window_samples(fs_hz)
        )

        if len(event_indices) == 0:
            return self._empty_candidate_set(fs_hz)

        event_values = self.analysis_trace[np.asarray(event_indices, dtype=int)]
        polarity = 1 if np.nanmedian(event_values) >= 0 else -1
        long_source = self.analysis_trace if self.mode == "slow" else self.denoiser_input

        keep_indices = []
        aligned = []
        amplitudes = []
        short_snippets = []
        long_snippets = []
        for idx in np.asarray(event_indices, dtype=int):
            search_lo = max(0, idx - align)
            search_hi = min(len(self.analysis_trace), idx + align + 1)
            if search_hi <= search_lo:
                continue
            peak = search_lo + int(
                np.argmax(polarity * self.analysis_trace[search_lo:search_hi])
            )
            short_lo = peak - short_pre
            short_hi = peak + short_post + 1
            long_lo = peak - long_pre
            long_hi = peak + long_post + 1
            if short_lo < 0 or short_hi > len(self.analysis_trace):
                continue

            short = polarity * self.analysis_trace[short_lo:short_hi].copy()
            short -= np.median(short[:short_pre])
            long = np.full(long_pre + long_post + 1, np.nan, dtype=float)
            src_lo = max(0, long_lo)
            src_hi = min(len(long_source), long_hi)
            dst_lo = src_lo - long_lo
            dst_hi = dst_lo + (src_hi - src_lo)
            baseline_lo = max(0, peak - long_pre)
            baseline_hi = max(baseline_lo + 1, peak)
            baseline = np.median(long_source[baseline_lo:baseline_hi])
            long[dst_lo:dst_hi] = polarity * (long_source[src_lo:src_hi] - baseline)

            keep_indices.append(idx)
            aligned.append(peak)
            amplitudes.append(short[short_pre])
            short_snippets.append(short)
            long_snippets.append(long)

        if not keep_indices:
            return self._empty_candidate_set(fs_hz)

        short_snippets = np.asarray(short_snippets, dtype=float)
        long_snippets = np.asarray(long_snippets, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)
        aligned = np.asarray(aligned, dtype=int)
        keep_indices = np.asarray(keep_indices, dtype=int)

        long_time_ms = (np.arange(long_snippets.shape[1]) - long_pre) / fs_hz * 1000.0
        template_time_ms = (np.arange(short_snippets.shape[1]) - short_pre) / fs_hz * 1000.0
        pca_scores, pca_explained_variance = candidate_pca_embedding(
            long_snippets,
            long_time_ms,
            window_ms=400.0,
        )

        return CandidateSet(
            indices=keep_indices,
            aligned_indices=aligned,
            times_s=aligned / fs_hz,
            amplitudes=amplitudes,
            short_snippets=short_snippets,
            long_snippets=long_snippets,
            long_time_ms=long_time_ms,
            template_time_ms=template_time_ms,
            pca_scores=pca_scores,
            pca_explained_variance=pca_explained_variance,
        )

    def _initial_seed_indices(self):
        if self.mode == "manual" or len(self.candidates.indices) == 0:
            return np.array([], dtype=int)
        eligible = np.flatnonzero(
            np.isin(self.candidates.indices, self.threshold_event_indices)
        )
        if len(eligible) == 0:
            return np.array([], dtype=int)
        n_seed = max(
            1,
            int(np.ceil(len(eligible) * (1.0 - self.high_amplitude_quantile))),
        )
        n_seed = min(len(eligible), n_seed)
        ranked = eligible[
            np.argsort(self.candidates.amplitudes[eligible])[::-1][:n_seed]
        ]
        return np.sort(ranked).astype(int)

    def _compute_initial_template_state(self):
        self.initial_template_source = np.asarray(self.seed_indices, dtype=int)
        self.initial_template_scores = np.full(len(self.candidates.indices), np.nan)
        if self.mode == "manual" or len(self.initial_template_source) == 0:
            return

        snippets = self.candidates.short_snippets[self.initial_template_source]
        if len(snippets) == 0:
            return
        initial_template = np.mean(snippets, axis=0)
        self.initial_template_scores = cosine_similarity_rows(
            self.candidates.short_snippets,
            initial_template,
        )

    def _build_widgets(self):
        self.path_text = widgets.Text(
            value=str(self.data_path),
            description="data path",
            layout=widgets.Layout(width="900px"),
        )
        self.scan_button = widgets.Button(
            description="Scan",
            layout=widgets.Layout(width="110px"),
        )
        self.scan_button.on_click(self._scan_data_path)

        self.animal_dropdown = widgets.Dropdown(
            options=[("Select animal", "")],
            value="",
            description="animal",
            layout=widgets.Layout(width=f"{PANEL_WIDTH_PX}px"),
        )
        self.experiment_dropdown = widgets.Dropdown(
            options=[("Select experiment", "")],
            value="",
            description="experiment",
            layout=widgets.Layout(width=f"{PANEL_WIDTH_PX}px"),
        )
        self.animal_dropdown.observe(self._animal_changed, names="value")
        self.experiment_dropdown.observe(self._experiment_changed, names="value")

        options = self._recording_options()
        self.recording_dropdown = widgets.Dropdown(
            options=options,
            value=options[0][1],
            description="recording",
            layout=widgets.Layout(width=f"{PANEL_WIDTH_PX}px"),
        )
        self.load_button = widgets.Button(
            description="Load",
            button_style="primary",
            layout=widgets.Layout(width="130px"),
        )
        self.load_button.on_click(self._load_selected_recording)
        self.view_filter = widgets.ToggleButtons(
            options=[("All", "all"), ("Yes", "yes"), ("No", "no"), ("Unlabeled", "unlabeled")],
            value="all",
            description="",
            layout=widgets.Layout(width="680px"),
        )
        self.view_filter.observe(lambda change: self._refresh_visible(), names="value")

        self.threshold_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            orientation="vertical",
            readout=True,
            readout_format=".2f",
            continuous_update=False,
            disabled=True,
            layout=widgets.Layout(width="90px", height="190px"),
        )
        self.threshold_slider.observe(self._threshold_changed, names="value")

        self.timeline_fig = go.FigureWidget()
        self.template_fig = go.FigureWidget()
        self.candidate_fig = go.FigureWidget()
        self.second_pass_fig = go.FigureWidget()
        self.feature_fig = go.FigureWidget()
        self.timeline_title = widgets.HTML()
        self.template_title = widgets.HTML()
        self.candidate_title = widgets.HTML()
        self.second_pass_title = widgets.HTML()
        self.feature_title = widgets.HTML()
        self.timeline_box = self._plot_panel(
            self.timeline_title,
            self.timeline_fig,
            TIMELINE_PLOT_WIDTH_PX,
            TIMELINE_HEIGHT_PX,
        )
        self.threshold_box = widgets.VBox(
            [
                widgets.HTML(self._panel_title("A1. Threshold")),
                widgets.HBox(
                    [self.threshold_slider],
                    layout=widgets.Layout(justify_content="center"),
                ),
            ],
            layout=widgets.Layout(
                border="1px solid #ddd",
                height=f"{TIMELINE_HEIGHT_PX + 26}px",
                padding="8px",
                width=f"{THRESHOLD_PANEL_WIDTH_PX}px",
            ),
        )
        self.template_box = self._plot_panel(
            self.template_title,
            self.template_fig,
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self.candidate_box = self._plot_panel(
            self.candidate_title,
            self.candidate_fig,
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self.second_pass_box = self._plot_panel(
            self.second_pass_title,
            self.second_pass_fig,
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self.feature_box = self._plot_panel(
            self.feature_title,
            self.feature_fig,
            PANEL_WIDTH_PX,
            CONTROL_PANEL_HEIGHT_PX,
        )

        decision_button_layout = {
            "width": f"{DECISION_BUTTON_WIDTH_PX}px",
            "min_width": f"{DECISION_BUTTON_WIDTH_PX}px",
            "max_width": f"{DECISION_BUTTON_WIDTH_PX}px",
            "height": f"{DECISION_BUTTON_HEIGHT_PX}px",
            "min_height": f"{DECISION_BUTTON_HEIGHT_PX}px",
            "max_height": f"{DECISION_BUTTON_HEIGHT_PX}px",
            "flex": f"0 0 {DECISION_BUTTON_WIDTH_PX}px",
            "overflow": "visible",
        }
        self.yes_button = widgets.Button(
            description="Yes",
            button_style="success",
            layout=widgets.Layout(**decision_button_layout),
        )
        self.no_button = widgets.Button(
            description="No",
            button_style="danger",
            layout=widgets.Layout(**decision_button_layout),
        )
        self.clear_button = widgets.Button(
            description="Clear",
            button_style="",
            layout=widgets.Layout(**decision_button_layout),
        )
        self.yes_button.add_class("vnoiser-decision-action")
        self.yes_button.add_class("vnoiser-decision-yes")
        self.no_button.add_class("vnoiser-decision-action")
        self.no_button.add_class("vnoiser-decision-no")
        self.clear_button.add_class("vnoiser-decision-action")
        self.clear_button.add_class("vnoiser-decision-clear")
        self.yes_button.on_click(lambda _: self._set_label("yes"))
        self.no_button.on_click(lambda _: self._set_label("no"))
        self.clear_button.on_click(lambda _: self._set_label("unlabeled"))
        self.decision_buttons = widgets.HBox(
            [self.yes_button, self.no_button, self.clear_button],
            layout=widgets.Layout(
                align_items="center",
                flex_flow="row nowrap",
                height=f"{DECISION_ROW_HEIGHT_PX}px",
                min_height=f"{DECISION_ROW_HEIGHT_PX}px",
                margin="6px 0 12px 0",
                overflow="visible",
                width=(
                    f"{DECISION_BUTTON_WIDTH_PX * 3 + DECISION_BUTTON_GAP_PX * 2}px"
                ),
            ),
        )
        self.decision_buttons.add_class("vnoiser-decision-row")
        self.current_info = widgets.HTML()
        self.save_info = widgets.HTML()
        self.save_info.add_class("vnoiser-wrap-path")
        self.decision_box = widgets.VBox(
            [
                widgets.HTML("<b>Decision</b>"),
                self.decision_buttons,
                self.current_info,
                self.save_info,
            ],
            layout=widgets.Layout(
                border="1px solid #ddd",
                min_height=f"{CONTROL_PANEL_HEIGHT_PX}px",
                overflow="visible",
                padding="10px",
                width=f"{PANEL_WIDTH_PX}px",
            ),
        )
        self.decision_box.add_class("vnoiser-control-panel")

        self.prev_button = widgets.Button(description="< Event", layout=widgets.Layout(width="110px"))
        self.next_button = widgets.Button(description="Event >", layout=widgets.Layout(width="110px"))
        self.prev_button.on_click(lambda _: self._step_event(-1))
        self.next_button.on_click(lambda _: self._step_event(1))
        self.event_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="event",
            continuous_update=False,
            layout=widgets.Layout(width="100%"),
        )
        self.event_slider.observe(self._slider_changed, names="value")
        self.nav_info = widgets.HTML()
        self.nav_box = widgets.VBox(
            [
                widgets.HTML("<b>Navigation</b>"),
                widgets.HBox([self.prev_button, self.next_button]),
                self.event_slider,
                self.nav_info,
            ],
            layout=widgets.Layout(
                border="1px solid #ddd",
                min_height=f"{CONTROL_PANEL_HEIGHT_PX}px",
                padding="10px",
                width=f"{PANEL_WIDTH_PX}px",
            ),
        )
        self.nav_box.add_class("vnoiser-control-panel")

        row = widgets.Layout(
            width=f"{DASHBOARD_WIDTH_PX}px",
            justify_content="space-between",
            align_items="flex-start",
            margin=f"0 0 {PANEL_GAP_PX}px 0",
        )
        second_row = widgets.HBox(
            [self.template_box, self.candidate_box, self.second_pass_box],
            layout=row,
        )
        third_row = widgets.HBox(
            [self.feature_box, self.decision_box, self.nav_box],
            layout=widgets.Layout(
                width=f"{DASHBOARD_WIDTH_PX}px",
                justify_content="space-between",
                align_items="flex-start",
            ),
        )
        timeline_row = widgets.HBox(
            [self.timeline_box, self.threshold_box],
            layout=widgets.Layout(
                width=f"{DASHBOARD_WIDTH_PX}px",
                justify_content="space-between",
                align_items="flex-start",
                margin=f"0 0 {PANEL_GAP_PX}px 0",
            ),
        )
        self.grid = widgets.VBox(
            [timeline_row, second_row, third_row],
            layout=widgets.Layout(width=f"{DASHBOARD_WIDTH_PX}px"),
        )

        path_controls = widgets.HBox(
            [self.path_text, self.scan_button],
            layout=widgets.Layout(
                align_items="center",
                margin=f"0 0 {PANEL_GAP_PX}px 0",
                width=f"{DASHBOARD_WIDTH_PX}px",
            ),
        )
        self.hierarchy_controls = widgets.HBox(
            [self.animal_dropdown, self.experiment_dropdown],
            layout=widgets.Layout(
                align_items="center",
                justify_content="space-between",
                margin=f"0 0 {PANEL_GAP_PX}px 0",
                width=f"{DASHBOARD_WIDTH_PX}px",
            ),
        )
        load_controls = widgets.HBox(
            [
                self.recording_dropdown,
                self.load_button,
                widgets.HTML("<b>view</b>", layout=widgets.Layout(width="42px")),
                self.view_filter,
            ],
            layout=widgets.Layout(
                align_items="center",
                justify_content="space-between",
                margin=f"0 0 {PANEL_GAP_PX}px 0",
                width=f"{DASHBOARD_WIDTH_PX}px",
            ),
        )
        title = {
            "manual": "Manual template curation",
            "fast": "Fast-event seeded template curation",
            "slow": "Slow-event seeded template curation (<40 Hz)",
        }[self.mode]
        self.status = widgets.HTML()
        self.dashboard_style = widgets.HTML(
            DECISION_BUTTON_STYLE_HTML,
            layout=widgets.Layout(height="0px", min_height="0px", overflow="hidden"),
        )
        self.ui = widgets.VBox(
            [
                self.dashboard_style,
                widgets.HTML(f"<h3>{title}</h3>"),
                path_controls,
                self.hierarchy_controls,
                load_controls,
                self.status,
                self.grid,
            ],
            layout=widgets.Layout(width=f"{DASHBOARD_WIDTH_PX}px", overflow="auto"),
        )
        self.ui.add_class("vnoiser-curation-dashboard")
        self._sync_dataset_controls()

    @staticmethod
    def _plot_panel(title_widget, fig, width_px, height_px):
        title_widget.layout = widgets.Layout(width=f"{width_px}px")
        plot_box = widgets.Box(
            [fig],
            layout=widgets.Layout(
                width=f"{width_px}px",
                height=f"{height_px}px",
                overflow="hidden",
            ),
        )
        return widgets.VBox(
            [title_widget, plot_box],
            layout=widgets.Layout(width=f"{width_px}px", overflow="hidden"),
        )

    @staticmethod
    def _panel_title(text):
        return (
            '<div style="font-weight:600;font-size:16px;'
            'line-height:22px;color:#263b5e;margin:0 0 4px 0;">'
            f"{text}</div>"
        )

    @staticmethod
    def _reset_figure(fig):
        fig.data = []
        fig.layout.shapes = []
        fig.layout.annotations = []

    def _set_loaded_controls(self, enabled):
        has_events = bool(enabled and len(self.event_keys))
        self.threshold_slider.disabled = not bool(enabled)
        for button in (self.yes_button, self.no_button, self.clear_button):
            button.disabled = not has_events
        self.prev_button.disabled = not has_events
        self.next_button.disabled = not has_events
        self.event_slider.disabled = not has_events

    def _draw_empty_all(self, message):
        self._draw_empty_plot(
            self.timeline_fig,
            self.timeline_title,
            "A. Full trace and candidate events",
            message,
            TIMELINE_PLOT_WIDTH_PX,
            TIMELINE_HEIGHT_PX,
        )
        self._draw_empty_plot(
            self.template_fig,
            self.template_title,
            "B. Current template",
            "No recording loaded.",
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self._draw_empty_plot(
            self.candidate_fig,
            self.candidate_title,
            "C. Focused candidate",
            "No candidate selected.",
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self._draw_empty_plot(
            self.second_pass_fig,
            self.second_pass_title,
            "D. Second pass preview",
            "No recording loaded.",
            PANEL_WIDTH_PX,
            PANEL_HEIGHT_PX,
        )
        self._draw_empty_plot(
            self.feature_fig,
            self.feature_title,
            "E. Candidate PCA (400 ms)",
            "No candidates loaded.",
            PANEL_WIDTH_PX,
            CONTROL_PANEL_HEIGHT_PX,
        )
        self.current_info.value = "No event loaded."
        self.save_info.value = (
            '<span class="vnoiser-wrap-path">curation file: '
            f"{self.label_path}</span>"
        )
        self.nav_info.value = (
            f"recordings found: {len(self.recordings)}<br>"
            f"cache: {self.curation_dir / 'cache'}"
        )
        self.status.value = f"<b>Status:</b> {message}"

    def _draw_empty_plot(self, fig, title_widget, title, message, width_px, height_px):
        self._reset_figure(fig)
        title_widget.value = self._panel_title(title)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False))
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(
            height=height_px,
            width=width_px,
            template="plotly_white",
            font={"size": 12},
            margin={"t": 18, "b": 38, "l": 55, "r": 18},
            showlegend=False,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    def _sync_dataset_controls(self):
        spatial = isinstance(self.dataset, SpatialJediDataset)
        hierarchical = spatial and self.dataset.requires_experiment_selection
        self._updating_dataset_controls = True
        try:
            self.hierarchy_controls.layout.display = "flex" if hierarchical else "none"
            if hierarchical:
                animals = self.dataset.animal_options()
                self.animal_dropdown.options = [("Select animal", ""), *animals]
                if self.dataset.scope == "animal" and animals:
                    animal_value = animals[0][1]
                    self.animal_dropdown.value = animal_value
                    experiments = self.dataset.experiment_options(animal_value)
                    self.experiment_dropdown.options = [
                        ("Select experiment", ""),
                        *experiments,
                    ]
                else:
                    self.animal_dropdown.value = ""
                    self.experiment_dropdown.options = [("Select experiment", "")]
                self.experiment_dropdown.value = ""
            else:
                self.animal_dropdown.options = [("Select animal", "")]
                self.experiment_dropdown.options = [("Select experiment", "")]
                self.animal_dropdown.value = ""
                self.experiment_dropdown.value = ""

            options = self._recording_options()
            self.recording_dropdown.options = options
            self.recording_dropdown.value = options[0][1]
            self.load_button.disabled = not bool(self.recordings)
        finally:
            self._updating_dataset_controls = False

    def _animal_changed(self, change):
        if self._updating_dataset_controls:
            return
        animal = change["new"]
        self._updating_dataset_controls = True
        try:
            experiments = self.dataset.experiment_options(animal) if animal else []
            self.experiment_dropdown.options = [
                ("Select experiment", ""),
                *experiments,
            ]
            self.experiment_dropdown.value = ""
            self.recordings = tuple()
            self.recording_dropdown.options = [("Select an experiment first", "")]
            self.recording_dropdown.value = ""
            self.load_button.disabled = True
        finally:
            self._updating_dataset_controls = False
        self._reset_loaded_selection("Select one experiment to load its scan metadata.")

    def _experiment_changed(self, change):
        if self._updating_dataset_controls:
            return
        experiment = change["new"]
        if not experiment:
            return

        self.status.value = "<b>Status:</b> loading metadata for one experiment..."
        try:
            self.recordings = self.dataset.select_experiment(experiment)
        except Exception as exc:
            self.recordings = tuple()
            self.recording_dropdown.options = [("No recordings found", "")]
            self.recording_dropdown.value = ""
            self.load_button.disabled = True
            self._reset_loaded_selection(f"Experiment unavailable: {exc}")
            return

        options = self._recording_options()
        self.recording_dropdown.options = options
        self.recording_dropdown.value = options[0][1]
        self.load_button.disabled = not bool(self.recordings)
        self._reset_loaded_selection(
            f"Loaded metadata for one experiment; choose from {len(self.recordings)} scan/domain traces."
        )

    def _reset_loaded_selection(self, message):
        self.recording = None
        self.event_keys = []
        self.visible_indices = np.array([], dtype=int)
        self.event_slider.max = 0
        self.event_slider.value = 0
        self._set_loaded_controls(False)
        self._draw_empty_all(message)

    def _scan_data_path(self, _):
        SpatialJediDataset.clear_inventory_cache()
        self.recording = None
        self._configure_data_path(self.path_text.value)
        self._sync_dataset_controls()
        self._reset_loaded_selection(self._initial_message())

    def _load_selected_recording(self, _):
        value = self.recording_dropdown.value
        if not value:
            self._draw_empty_all("No recording is selected.")
            return

        self.status.value = "<b>Status:</b> loading one selected recording..."
        try:
            if isinstance(self.dataset, SpatialJediDataset):
                full = self.dataset.load(value, load_events=False)
            else:
                full = self.dataset.load(value)
            recording = self._window_from_recording(full)
            self._activate_recording_storage(recording)
            self._run_or_load_pipeline(recording)
        except Exception as exc:
            self.recording = None
            self.event_keys = []
            self.visible_indices = np.array([], dtype=int)
            self._set_loaded_controls(False)
            self._draw_empty_all(f"Load failed: {exc}")
            raise

        self.event_slider.max = max(0, len(self.candidates.indices) - 1)
        self.event_slider.value = 0
        self._set_loaded_controls(True)
        self._refresh_all()
        self.status.value = (
            f"<b>Status:</b> loaded {self.recording.metadata.get('label', self.recording.path.name)}; "
            f"{len(self.event_keys)} candidates; {self.pipeline_cache_status}."
        )

    def _template_indices(self):
        labels = [self.labels.get(key, "unlabeled") for key in self.event_keys]
        yes_indices = [i for i, label in enumerate(labels) if label == "yes"]
        if self.mode == "manual":
            return np.asarray(yes_indices, dtype=int)
        seed_indices = [
            int(i)
            for i in self.seed_indices
            if i < len(labels) and labels[int(i)] != "no"
        ]
        return np.asarray(sorted(set(seed_indices + yes_indices)), dtype=int)

    def _compute_template_state(self):
        self.template_source = self._template_indices()
        if len(self.template_source) == 0:
            self.template = None
            self.template_scores = np.full(len(self.candidates.indices), np.nan)
            return
        snippets = self.candidates.short_snippets[self.template_source]
        self.template = np.mean(snippets, axis=0)
        self.template_scores = cosine_similarity_rows(
            self.candidates.short_snippets,
            self.template,
        )

    def _compute_second_pass(self):
        self.second_pass = self.denoised.copy()
        rejected = [
            i for i, key in enumerate(self.event_keys)
            if self.labels.get(key, "unlabeled") == "no"
        ]
        fs_hz = self.recording.fs_hz
        if self.mode == "slow":
            pre = max(1, int(round(0.500 * fs_hz)))
            post = max(1, int(round(0.500 * fs_hz)))
        else:
            pre = max(1, int(round(0.004 * fs_hz)))
            post = max(1, int(round(0.012 * fs_hz)))
        for i in rejected:
            peak = int(self.candidates.aligned_indices[i])
            lo = max(0, peak - pre)
            hi = min(len(self.second_pass), peak + post + 1)
            if hi - lo < 3:
                continue
            self.second_pass[lo:hi] = np.linspace(
                self.second_pass[lo],
                self.second_pass[hi - 1],
                hi - lo,
            )

    def _refresh_all(self):
        if self.recording is None:
            self._draw_empty_all(self._initial_message())
            return
        self._compute_template_state()
        self._compute_second_pass()
        self._refresh_visible(reselect=False, redraw=False)
        self._draw_timeline()
        self._draw_template()
        self._draw_candidate()
        self._draw_second_pass()
        self._draw_features()
        self._refresh_status()

    def _refresh_visible(self, reselect=True, redraw=True):
        if self.recording is None:
            return

        label_filter = self.view_filter.value
        if label_filter == "all":
            visible = np.arange(len(self.candidates.indices), dtype=int)
        else:
            visible = np.asarray(
                [
                    i for i, key in enumerate(self.event_keys)
                    if self.labels.get(key, "unlabeled") == label_filter
                ],
                dtype=int,
            )
        self.visible_indices = visible
        if reselect and len(visible) and self.current not in set(visible.tolist()):
            self.current = int(visible[0])
        if redraw:
            self._draw_timeline()
            self._draw_features()
            self._refresh_status()

    def _label_for_index(self, i):
        label = self.labels.get(self.event_keys[i], "unlabeled")
        if label != "unlabeled":
            return label
        initial_call = self._initial_auto_call_for_index(i)
        if initial_call == "pass":
            return "auto_yes"
        if initial_call == "reject":
            return "auto_no"
        return label

    def _initial_template_score_for_index(self, i):
        if self.mode == "manual" or len(self.initial_template_scores) <= i:
            return np.nan
        return self.initial_template_scores[i]

    def _initial_auto_call_for_index(self, i):
        score = self._initial_template_score_for_index(i)
        if not np.isfinite(score):
            return None
        return "pass" if score > AUTO_TEMPLATE_THRESHOLD else "reject"

    def _marker_colors(self, indices: Iterable[int]):
        return [LABEL_COLORS[self._label_for_index(int(i))] for i in indices]

    def _draw_timeline(self):
        self._reset_figure(self.timeline_fig)
        candidate_kind = "slow (<40 Hz)" if self.mode == "slow" else "fast"
        self.timeline_title.value = self._panel_title(
            f"A. Full trace and {candidate_kind} candidate events"
        )
        t_plot, input_plot = downsample_xy(self.recording.t, self.denoised, 5000)
        self.timeline_fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=input_plot,
                mode="lines",
                name="denoised trace",
                showlegend=True,
                line={"color": "#111111", "width": 1},
            )
        )
        marker_source = self.denoised
        if self.mode == "slow":
            slow_t, slow_plot = downsample_xy(
                self.recording.t,
                self.analysis_trace,
                5000,
            )
            self.timeline_fig.add_trace(
                go.Scatter(
                    x=slow_t,
                    y=slow_plot,
                    mode="lines",
                    name="<40 Hz low-pass trace",
                    showlegend=True,
                    line={"color": "#2b6cb0", "width": 1.7},
                )
            )
            marker_source = self.analysis_trace
        self.timeline_fig.add_trace(
            go.Scatter(
                x=[float(self.recording.t[0]), float(self.recording.t[-1])],
                y=[self.candidate_threshold, self.candidate_threshold],
                mode="lines",
                name="candidate threshold",
                showlegend=True,
                line={"color": "#d7263d", "width": 1.5, "dash": "dash"},
                hovertemplate="threshold %{y:.2f}<extra></extra>",
            )
        )
        visible = self.visible_indices
        self.timeline_trace_indices = visible
        event_times = self.candidates.times_s[visible]
        event_y = np.interp(event_times, self.recording.t, marker_source)
        candidate_trace = go.Scatter(
            x=event_times,
            y=event_y,
            mode="markers",
            name=f"{candidate_kind} candidates",
            showlegend=True,
            marker={
                "color": self._marker_colors(visible),
                "size": 8,
                "line": {"color": "#222", "width": 0.4},
            },
            customdata=visible,
            hovertemplate="candidate %{customdata}<br>%{x:.3f} s<extra></extra>",
        )
        self.timeline_fig.add_trace(candidate_trace)
        if len(self.candidates.indices):
            current_t = self.candidates.times_s[self.current]
            current_y = np.interp(current_t, self.recording.t, marker_source)
            self.timeline_fig.add_trace(
                go.Scatter(
                    x=[current_t],
                    y=[current_y],
                    mode="markers",
                    name="focused event",
                    showlegend=False,
                    marker={
                        "color": "rgba(0,0,0,0)",
                        "size": 18,
                        "line": {"color": "#d7263d", "width": 3},
                    },
                    hoverinfo="skip",
                )
            )
            self.timeline_fig.add_vline(x=current_t, line_color="#d7263d", line_dash="dot")
        self.timeline_fig.update_layout(
            height=TIMELINE_HEIGHT_PX,
            width=TIMELINE_PLOT_WIDTH_PX,
            template="plotly_white",
            font={"size": 12},
            hovermode="x unified",
            margin={"t": 28, "b": 38, "l": 55, "r": 18},
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
                "bgcolor": "rgba(255,255,255,0.82)",
            },
        )
        self.timeline_fig.update_xaxes(
            title_text="time (s)",
            range=[float(self.recording.t[0]), float(self.recording.t[-1])],
            visible=True,
        )
        self.timeline_fig.update_yaxes(title_text="z", autorange=True, visible=True)
        candidate_trace.on_click(self._timeline_clicked)

    def _draw_template(self):
        self._reset_figure(self.template_fig)
        self.template_title.value = self._panel_title("B. Current template")
        if self.template is not None:
            for snippet in self.candidates.short_snippets[self.template_source]:
                self.template_fig.add_trace(
                    go.Scatter(
                        x=self.candidates.template_time_ms,
                        y=snippet,
                        mode="lines",
                        line={"color": "#4c78a8", "width": 1},
                        opacity=0.18,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            self.template_fig.add_trace(
                go.Scatter(
                    x=self.candidates.template_time_ms,
                    y=self.template,
                    mode="lines",
                    name="template",
                    line={"color": "#111111", "width": 3},
                )
            )
        else:
            self.template_fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="template"))
            self.template_fig.add_annotation(
                text="No Yes events yet",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        self.template_fig.update_layout(
            height=PANEL_HEIGHT_PX,
            width=PANEL_WIDTH_PX,
            template="plotly_white",
            font={"size": 12},
            margin={"t": 18, "b": 42, "l": 55, "r": 14},
            showlegend=False,
        )
        self.template_fig.update_xaxes(
            title_text="aligned time (ms)",
            range=[
                float(self.candidates.template_time_ms[0]),
                float(self.candidates.template_time_ms[-1]),
            ],
            visible=True,
        )
        self.template_fig.update_yaxes(
            title_text="baseline-subtracted z",
            autorange=True,
            visible=True,
        )

    def _draw_candidate(self):
        self._reset_figure(self.candidate_fig)
        if not len(self.candidates.indices):
            self._draw_empty_plot(
                self.candidate_fig,
                self.candidate_title,
                "C. Focused candidate",
                "No candidate events found.",
                PANEL_WIDTH_PX,
                PANEL_HEIGHT_PX,
            )
            return

        i = self.current
        candidate = self.candidates.long_snippets[i]
        x, y = downsample_xy(self.candidates.long_time_ms, candidate, 2500)
        self.candidate_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="candidate",
                line={"color": "#4c78a8", "width": 1.2},
            )
        )
        if self.template is not None:
            self.candidate_fig.add_trace(
                go.Scatter(
                    x=self.candidates.template_time_ms,
                    y=self.template,
                    mode="lines",
                    name="template",
                    line={"color": "#111111", "width": 2.5},
                )
            )
            sim_text = f"cosine {self.template_scores[i]:.2f}"
        else:
            sim_text = "cosine n/a"
        self.candidate_title.value = self._panel_title(f"C. Focused candidate ({sim_text})")
        self.candidate_fig.add_vline(x=0, line_dash="dot", line_color="#d7263d")
        self.candidate_fig.update_layout(
            height=PANEL_HEIGHT_PX,
            width=PANEL_WIDTH_PX,
            template="plotly_white",
            font={"size": 12},
            margin={"t": 24, "b": 42, "l": 55, "r": 14},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        self.candidate_fig.update_xaxes(
            title_text="time from event (ms)",
            range=[-500, 500] if self.mode == "slow" else [-100, 100],
            visible=True,
        )
        self.candidate_fig.update_yaxes(title_text="z", autorange=True, visible=True)

    def _draw_second_pass(self):
        self._reset_figure(self.second_pass_fig)
        if not len(self.candidates.indices):
            self._draw_empty_plot(
                self.second_pass_fig,
                self.second_pass_title,
                "D. Second pass preview",
                "No candidate events found.",
                PANEL_WIDTH_PX,
                PANEL_HEIGHT_PX,
            )
            return

        self.second_pass_title.value = self._panel_title("D. Second pass preview")
        i = self.current
        peak = int(self.candidates.aligned_indices[i])
        fs_hz = self.recording.fs_hz
        radius = max(1, int(round(0.500 * fs_hz)))
        lo = max(0, peak - radius)
        hi = min(len(self.denoised), peak + radius + 1)
        x_ms = (np.arange(lo, hi) - peak) / fs_hz * 1000.0
        x, y0 = downsample_xy(x_ms, self.denoised[lo:hi], 2500)
        _, y1 = downsample_xy(x_ms, self.second_pass[lo:hi], 2500)
        self.second_pass_fig.add_trace(
            go.Scatter(
                x=x,
                y=y0,
                mode="lines",
                name="original denoised",
                line={"color": "#9ecae9", "width": 1.1},
            )
        )
        self.second_pass_fig.add_trace(
            go.Scatter(
                x=x,
                y=y1,
                mode="lines",
                name="rejected removed",
                line={"color": "#d7263d", "width": 1.4},
            )
        )
        self.second_pass_fig.add_vline(x=0, line_dash="dot", line_color="#d7263d")
        self.second_pass_fig.update_layout(
            height=PANEL_HEIGHT_PX,
            width=PANEL_WIDTH_PX,
            template="plotly_white",
            font={"size": 12},
            margin={"t": 24, "b": 42, "l": 55, "r": 14},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        self.second_pass_fig.update_xaxes(
            title_text="time from event (ms)",
            range=[-500, 500],
            visible=True,
        )
        self.second_pass_fig.update_yaxes(title_text="z", autorange=True, visible=True)

    def _draw_features(self):
        self._reset_figure(self.feature_fig)
        if not len(self.candidates.indices):
            self._draw_empty_plot(
                self.feature_fig,
                self.feature_title,
                "E. Candidate PCA (400 ms)",
                "No candidate events found.",
                PANEL_WIDTH_PX,
                CONTROL_PANEL_HEIGHT_PX,
            )
            return

        self.feature_title.value = self._panel_title("E. Candidate PCA (400 ms)")
        scores = self.candidates.pca_scores
        explained = self.candidates.pca_explained_variance
        visible = self.visible_indices
        self.feature_fig.add_trace(
            go.Scatter(
                x=scores[visible, 0],
                y=scores[visible, 1],
                mode="markers",
                name="events",
                marker={
                    "color": self._marker_colors(visible),
                    "size": 8,
                    "opacity": 0.75,
                    "line": {"color": "#222", "width": 0.4},
                },
                customdata=visible + 1,
                hovertemplate=(
                    "event %{customdata}<br>PC1 %{x:.2f}<br>"
                    "PC2 %{y:.2f}<extra></extra>"
                ),
            )
        )
        self.feature_fig.add_trace(
            go.Scatter(
                x=[scores[self.current, 0]],
                y=[scores[self.current, 1]],
                mode="markers",
                name="focused",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 17,
                    "line": {"color": "#d7263d", "width": 3},
                },
                hoverinfo="skip",
            )
        )
        self.feature_fig.update_layout(
            height=CONTROL_PANEL_HEIGHT_PX,
            width=PANEL_WIDTH_PX,
            template="plotly_white",
            font={"size": 12},
            margin={"t": 18, "b": 42, "l": 55, "r": 14},
            showlegend=False,
        )
        self.feature_fig.update_xaxes(
            title_text=f"PC1 ({explained[0] * 100.0:.1f}%)",
            autorange=True,
            visible=True,
        )
        self.feature_fig.update_yaxes(
            title_text=f"PC2 ({explained[1] * 100.0:.1f}%)",
            autorange=True,
            visible=True,
        )
        self.feature_fig.data[0].on_click(self._feature_clicked)

    def _refresh_status(self):
        if self.recording is None or not self.event_keys:
            self._set_loaded_controls(self.recording is not None)
            self.current_info.value = "No event loaded."
            self.save_info.value = f"curation file: {self.label_path}"
            self.nav_info.value = f"recordings found: {len(self.recordings)}"
            return

        labels = [self.labels.get(key, "unlabeled") for key in self.event_keys]
        n_yes = labels.count("yes")
        n_no = labels.count("no")
        n_unlabeled = labels.count("unlabeled")
        label = self.labels.get(self.event_keys[self.current], "unlabeled")
        score = self.template_scores[self.current] if len(self.template_scores) else np.nan
        score_text = "n/a" if not np.isfinite(score) else f"{score:.2f}"
        auto_text = ""
        initial_score = self._initial_template_score_for_index(self.current)
        initial_call = self._initial_auto_call_for_index(self.current)
        candidate_source = (
            "threshold detection"
            if int(self.candidates.indices[self.current])
            in set(self.threshold_event_indices.tolist())
            else "retained manual event"
        )
        if self.mode in {"fast", "slow"} and initial_call is not None:
            initial_score_text = f"{initial_score:.2f}"
            auto_text = (
                f"<br>initial template: <b>{initial_call}</b> "
                f"({initial_score_text}) at {AUTO_TEMPLATE_THRESHOLD:.2f}"
            )
        self.current_info.value = (
            f"<b>event {self.current + 1}/{len(self.event_keys)}</b><br>"
            f"label: <b>{label}</b><br>"
            f"time: {self.candidates.times_s[self.current]:.3f} s<br>"
            f"source: {candidate_source}<br>"
            f"template cosine: {score_text}"
            f"{auto_text}"
        )
        self.save_info.value = (
            f"yes {n_yes} | no {n_no} | unlabeled {n_unlabeled}<br>"
            '<span class="vnoiser-wrap-path">curation file: '
            f"{self.label_path}</span>"
        )
        self.nav_info.value = (
            f"recording: {self.recording.metadata.get('label', self.recording.path.name)}<br>"
            f"candidates: {len(self.event_keys)} | visible: {len(self.visible_indices)}<br>"
            f"template source events: {len(self.template_source)}<br>"
            f"cache: {self.pipeline_cache_status}"
        )
        if self.event_slider.value != self.current:
            self.event_slider.unobserve(self._slider_changed, names="value")
            self.event_slider.value = self.current
            self.event_slider.observe(self._slider_changed, names="value")

    def _timeline_clicked(self, trace, points, state):
        if not points.point_inds:
            return
        event_idx = int(self.timeline_trace_indices[points.point_inds[0]])
        self._select_event(event_idx)

    def _feature_clicked(self, trace, points, state):
        if not points.point_inds:
            return
        event_idx = int(self.visible_indices[points.point_inds[0]])
        self._select_event(event_idx)

    def _slider_changed(self, change):
        self._select_event(int(change["new"]))

    def _step_event(self, step):
        if len(self.visible_indices) == 0:
            return
        visible = self.visible_indices.tolist()
        if self.current in visible:
            pos = visible.index(self.current)
        else:
            pos = 0
        next_pos = (pos + step) % len(visible)
        self._select_event(int(visible[next_pos]))

    def _select_event(self, event_idx):
        if event_idx < 0 or event_idx >= len(self.event_keys):
            return
        self.current = int(event_idx)
        self._draw_timeline()
        self._draw_candidate()
        self._draw_second_pass()
        self._draw_features()
        self._refresh_status()

    def _set_label(self, label):
        if not self.event_keys:
            return
        current_key = self.event_keys[self.current]
        if label == "unlabeled":
            self.labels.pop(current_key, None)
        else:
            self.labels[current_key] = label
        self._compute_template_state()
        self._compute_second_pass()
        self._save_labels()
        if label == "unlabeled":
            self._rebuild_candidates(preserve_key=current_key)
            self._set_loaded_controls(True)
            self._refresh_all()
            return
        self._refresh_visible(reselect=False)
        self._draw_template()
        self._draw_candidate()
        self._draw_second_pass()
        self._draw_features()
        self._refresh_status()
