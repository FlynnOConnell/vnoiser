"""Peaks on the final denoised trace: stage 5 of the spatial JEDI archive.

The notebook (``Analysis_PF_2.ipynb``) band-passed the trace, took the local
maxima above a threshold on the band-passed trace, dropped those whose
height on the trace itself was below an amplitude threshold, and dropped
those whose above-baseline stretch was too short. Its ``param_spike_detect.pkl``
holds the thresholds; the peak spacing and the on/off level were literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


@dataclass
class SpikeDetectConfig:
    """Thresholds of :func:`detect_peaks`.

    ``bp``, ``thres_bp_sd``, ``thres_amp_sd`` and ``duration_thres_ms`` are
    the archive's ``param_spike_detect.pkl`` keys (``duration_thres`` there,
    in ms). ``distance_samples`` and ``onoff_sd`` were literals in the
    notebook; 3 samples reproduces the saved peaks where anything does.
    """

    bp: tuple = (2.0, 400.0)
    thres_bp_sd: float = 3.5
    thres_amp_sd: float = 4.0
    duration_thres_ms: float = 5.0
    distance_samples: int = 3
    order: int = 5
    onoff_sd: float = 0.5

    def to_param_pickle(self) -> dict:
        """The ``param_spike_detect.pkl`` payload."""
        return {
            "thres_bp_sd": self.thres_bp_sd,
            "thres_amp_sd": self.thres_amp_sd,
            "bp": [float(self.bp[0]), float(self.bp[1])],
            "duration_thres": self.duration_thres_ms,
        }

    @classmethod
    def from_param_pickle(cls, params: dict, **overrides) -> SpikeDetectConfig:
        """From a ``param_spike_detect.pkl`` payload."""
        fields = dict(
            bp=tuple(float(v) for v in params["bp"]),
            thres_bp_sd=float(params["thres_bp_sd"]),
            thres_amp_sd=float(params["thres_amp_sd"]),
            duration_thres_ms=float(params["duration_thres"]),
        )
        fields.update(overrides)
        return cls(**fields)


def detect_peaks(trace, fs: float, cfg: SpikeDetectConfig | None = None) -> np.ndarray:
    """Event peaks of one denoised trace, sorted sample indices.

    Parameters
    ----------
    trace : array (n_frame,)
        The final denoised trace (``denoised_trace_scans.pkl``).
    fs : float
        Sampling rate in Hz. The archive ran this at ``floor(fs)``, the
        value in ``fs_scans.pkl``.
    cfg : SpikeDetectConfig, optional
    """
    cfg = cfg or SpikeDetectConfig()
    y = np.asarray(trace, dtype=float).ravel()
    if y.size < 3 * (cfg.order + 1):
        return np.array([], dtype=int)
    nyq = 0.5 * fs
    b, a = butter(cfg.order, [cfg.bp[0] / nyq, cfg.bp[1] / nyq], btype="band")
    y_bp = filtfilt(b, a, y)
    height = np.mean(y_bp) + cfg.thres_bp_sd * np.std(y_bp)
    idx, _ = find_peaks(y_bp, height=height, distance=max(1, int(cfg.distance_samples)))
    idx = idx[y[idx] >= np.mean(y) + cfg.thres_amp_sd * np.std(y)]
    # the stretch above median + onoff_sd * sd around each peak must last duration_thres_ms
    below = y < np.median(y) + np.std(y) * cfg.onoff_sd
    min_samples = cfg.duration_thres_ms / 1000.0 * fs
    keep = np.ones(idx.size, dtype=bool)
    for k, i in enumerate(idx):
        before = np.flatnonzero(below[:i])
        after = np.flatnonzero(below[i:])
        if before.size and after.size and (after[0] + i - before[-1]) < min_samples:
            keep[k] = False
    return np.sort(idx[keep]).astype(int)
