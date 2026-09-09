"""Simulation utilities for voltage-imaging denoiser development."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Jedi3SubConfig:
    """Parameters for a noisy 1D JEDI3sub-like linescan simulation.

    The simulator is independent of any denoiser. It returns voltage and
    bright-to-dim JEDI3sub fluorescence, where depolarizing events produce
    negative dF/F deflections. Downstream code can sign-flip or otherwise
    transform the trace if a particular detector expects another convention.
    """

    fs_hz: float = 10000.0
    resting_mv: float = -70.0
    isolated_ap_rate_hz: float = 6.0
    fast_event_rate_hz: float = 24.0
    slow_event_rate_hz: float = 1.2
    burst_rate_hz: float = 1.2
    ipsp_rate_hz: float = 7.0
    ripple_rate_hz: float = 0.6
    ap_peak_delta_mv: float = 100.0
    ap_afterhyperpolarization_mv: float = -8.0
    fast_event_mean_mv: float = 3.0
    fast_event_sd_mv: float = 1.0
    slow_event_mean_mv: float = 7.0
    slow_event_sd_mv: float = 2.0
    slow_event_duration_ms: float = 180.0
    burst_spike_count_min: int = 3
    burst_spike_count_max: int = 7
    burst_frequency_hz: float = 110.0
    burst_plateau_mean_mv: float = 8.0
    burst_plateau_sd_mv: float = 2.5
    ipsp_mean_mv: float = -3.5
    ipsp_sd_mv: float = 1.2
    ripple_amplitude_mv: float = 1.2
    ripple_frequency_hz: float = 160.0
    ripple_duration_ms: float = 90.0
    membrane_noise_sd_mv: float = 0.45
    slow_drift_mv: float = 1.0
    full_depol_dff: float = -0.90
    depol_reference_mv: float = 100.0
    depol_response_scale_mv: float = 55.0
    full_hyperpol_dff: float = 0.30
    hyperpol_reference_mv: float = 30.0
    hyperpol_response_scale_mv: float = 35.0
    indicator_delay_ms: float = 3.5
    depol_tau_fast_ms: float = 0.63
    depol_tau_slow_ms: float = 5.31
    depol_fast_fraction: float = 0.908
    repol_tau_ms: float = 2.23
    photon_rate_hz: float = 8.0e7
    read_noise_photons: float = 80.0
    electronic_noise_dff: float = 0.008
    scanline_gain_sd: float = 0.18
    motion_artifact_dff: float = 0.08
    ap_artifact_amplitude: float = 2.20
    ap_artifact_sd: float = 0.45
    ap_artifact_width_ms: float = 1.50
    ap_artifact_afterpulse: float = -0.08
    bleach_fraction: float = 0.05
    bleach_tau_s: float = 5.0


@dataclass
class SimulationResult:
    """Container returned by :func:`simulate_jedi3sub_trace`."""

    t: np.ndarray
    voltage_mv: np.ndarray
    dff_clean: np.ndarray
    dff_observed: np.ndarray
    dff_z: np.ndarray
    events: dict[str, Any]
    metadata: dict[str, Any]


def simulate_jedi3sub_trace(
    duration_s: float,
    config: Optional[Jedi3SubConfig] = None,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Simulate a noisy 1D JEDI3sub voltage-imaging trace.

    Parameters
    ----------
    duration_s:
        Simulated recording duration in seconds.
    config:
        Simulation parameters. If omitted, :class:`Jedi3SubConfig` is used.
    seed:
        Optional random seed for deterministic simulations.

    Returns
    -------
    SimulationResult
        Time axis, membrane voltage, clean dF/F, observed noisy dF/F,
        z-scored observed dF/F, biological event annotations, and metadata.
    """
    cfg = config or Jedi3SubConfig()
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if cfg.fs_hz <= 0:
        raise ValueError("config.fs_hz must be positive")

    rng = np.random.default_rng(seed)
    n_frame = int(round(duration_s * cfg.fs_hz))
    if n_frame < 2:
        raise ValueError("duration_s and fs_hz produce fewer than two samples")

    dt = 1.0 / cfg.fs_hz
    t = np.arange(n_frame, dtype=float) * dt

    voltage = np.full(n_frame, cfg.resting_mv, dtype=float)
    voltage += _slow_voltage_drift(t, cfg, rng)
    voltage += rng.normal(0.0, cfg.membrane_noise_sd_mv, size=n_frame)

    isolated_ap_times = _sample_event_times(
        cfg.isolated_ap_rate_hz,
        duration_s,
        rng,
        refractory_s=0.012,
    )
    fast_times = _sample_event_times(
        cfg.fast_event_rate_hz,
        duration_s,
        rng,
        refractory_s=0.003,
    )
    slow_times = _sample_event_times(
        cfg.slow_event_rate_hz,
        duration_s,
        rng,
        refractory_s=0.080,
    )
    burst_times = _sample_event_times(
        cfg.burst_rate_hz,
        duration_s,
        rng,
        refractory_s=0.120,
    )
    ipsp_times = _sample_event_times(cfg.ipsp_rate_hz, duration_s, rng, refractory_s=0.004)
    ripple_times = _sample_event_times(cfg.ripple_rate_hz, duration_s, rng, refractory_s=0.080)

    _add_events(voltage, isolated_ap_times, cfg.fs_hz, _ap_kernel(cfg))
    _add_fast_depolarizations(voltage, fast_times, cfg, rng)
    _add_slow_depolarizations(voltage, slow_times, cfg, rng)
    burst_spike_times = _add_bursts(voltage, burst_times, duration_s, cfg, rng)
    _add_hyperpolarizations(voltage, ipsp_times, cfg, rng)
    _add_ripples(voltage, ripple_times, cfg, rng)

    all_ap_times = np.sort(np.concatenate((isolated_ap_times, burst_spike_times)))
    dff_steady = _delay_trace(_voltage_to_steady_dff(voltage, cfg), cfg.indicator_delay_ms, cfg.fs_hz)
    dff_clean = _apply_jedi3sub_kinetics(dff_steady, cfg)
    dff_observed = _apply_linescan_observation_noise(dff_clean, t, cfg, rng)
    dff_observed += _ap_artifact_trace(n_frame, all_ap_times, cfg, rng)

    dff_std = np.std(dff_observed)
    if dff_std == 0:
        dff_z = np.zeros_like(dff_observed)
    else:
        dff_z = (dff_observed - np.mean(dff_observed)) / dff_std

    events = {
        "isolated_ap_times_s": isolated_ap_times,
        "isolated_ap_indices": _times_to_indices(isolated_ap_times, cfg.fs_hz, n_frame),
        "burst_times_s": burst_times,
        "burst_indices": _times_to_indices(burst_times, cfg.fs_hz, n_frame),
        "burst_ap_times_s": burst_spike_times,
        "burst_ap_indices": _times_to_indices(burst_spike_times, cfg.fs_hz, n_frame),
        "ap_times_s": all_ap_times,
        "ap_indices": _times_to_indices(all_ap_times, cfg.fs_hz, n_frame),
        "fast_depol_times_s": fast_times,
        "fast_depol_indices": _times_to_indices(fast_times, cfg.fs_hz, n_frame),
        "slow_depol_times_s": slow_times,
        "slow_depol_indices": _times_to_indices(slow_times, cfg.fs_hz, n_frame),
        "ipsp_times_s": ipsp_times,
        "ipsp_indices": _times_to_indices(ipsp_times, cfg.fs_hz, n_frame),
        "ripple_times_s": ripple_times,
        "ripple_indices": _times_to_indices(ripple_times, cfg.fs_hz, n_frame),
    }
    metadata = {
        "duration_s": duration_s,
        "fs_hz": cfg.fs_hz,
        "indicator": "JEDI3sub",
        "polarity": "bright-to-dim; depolarization and APs are negative dF/F deflections",
        "noise_model": "shot noise, read noise, electronic noise, scanline gain noise, motion artifact, bleaching",
        "config": asdict(cfg),
        "sources": [
            "https://experiments.springernature.com/articles/10.1038/s41592-026-03043-8",
            "https://www.researchgate.net/publication/403740874_Designer_indicators_for_two-photon_recording_of_subthreshold_voltage_dynamics",
            "https://www.sciencedirect.com/science/article/pii/S0092867419312255",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10156610/",
        ],
    }
    return SimulationResult(
        t=t,
        voltage_mv=voltage,
        dff_clean=dff_clean,
        dff_observed=dff_observed,
        dff_z=dff_z,
        events=events,
        metadata=metadata,
    )


def _sample_event_times(
    rate_hz: float,
    duration_s: float,
    rng: np.random.Generator,
    refractory_s: float,
) -> np.ndarray:
    if rate_hz <= 0:
        return np.array([], dtype=float)

    times = []
    t = rng.exponential(1.0 / rate_hz)
    while t < duration_s:
        if not times or t - times[-1] >= refractory_s:
            times.append(t)
        t += rng.exponential(1.0 / rate_hz)
    return np.asarray(times, dtype=float)


def _times_to_indices(times_s: np.ndarray, fs_hz: float, n_frame: int) -> np.ndarray:
    if len(times_s) == 0:
        return np.array([], dtype=int)
    return np.clip(np.rint(times_s * fs_hz).astype(int), 0, n_frame - 1)


def _slow_voltage_drift(
    t: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    phase_a = rng.uniform(0.0, 2.0 * np.pi)
    phase_b = rng.uniform(0.0, 2.0 * np.pi)
    drift = cfg.slow_drift_mv * (
        0.65 * np.sin(2.0 * np.pi * 0.35 * t + phase_a)
        + 0.35 * np.sin(2.0 * np.pi * 1.1 * t + phase_b)
    )
    wander = rng.normal(0.0, 0.03, size=len(t)).cumsum()
    wander -= np.mean(wander)
    max_abs = np.max(np.abs(wander))
    if max_abs > 0:
        wander = 0.6 * cfg.slow_drift_mv * wander / max_abs
    return drift + wander


def _ap_kernel(cfg: Jedi3SubConfig) -> np.ndarray:
    dt = 1.0 / cfg.fs_hz
    t = np.arange(0.0, 0.010, dt)
    depol = cfg.ap_peak_delta_mv * np.exp(-0.5 * ((t - 0.0012) / 0.00034) ** 2)
    ahp = cfg.ap_afterhyperpolarization_mv * np.exp(-0.5 * ((t - 0.0030) / 0.0012) ** 2)
    return depol + ahp


def _alpha_kernel(fs_hz: float, tau_rise_ms: float, tau_decay_ms: float) -> np.ndarray:
    dt = 1.0 / fs_hz
    duration_s = 8.0 * tau_decay_ms / 1000.0
    t = np.arange(0.0, duration_s, dt)
    tau_rise_s = tau_rise_ms / 1000.0
    tau_decay_s = tau_decay_ms / 1000.0
    kernel = np.exp(-t / tau_decay_s) - np.exp(-t / tau_rise_s)
    peak = np.max(np.abs(kernel))
    if peak > 0:
        kernel = kernel / peak
    return kernel


def _raised_cosine_plateau(fs_hz: float, duration_ms: float) -> np.ndarray:
    n = max(3, int(round(duration_ms * fs_hz / 1000.0)))
    edge = max(1, min(n // 4, int(round(0.020 * fs_hz))))
    plateau = np.ones(n, dtype=float)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge))
    plateau[:edge] = ramp
    plateau[-edge:] = ramp[::-1]
    return plateau


def _add_events(
    signal: np.ndarray,
    event_times_s: np.ndarray,
    fs_hz: float,
    kernel: np.ndarray,
) -> None:
    for event_t in event_times_s:
        start = int(round(event_t * fs_hz))
        if start >= len(signal):
            continue
        stop = min(start + len(kernel), len(signal))
        signal[start:stop] += kernel[: stop - start]


def _add_fast_depolarizations(
    voltage: np.ndarray,
    event_times_s: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> None:
    if len(event_times_s) == 0:
        return
    kernel = _alpha_kernel(cfg.fs_hz, tau_rise_ms=0.8, tau_decay_ms=12.0)
    amplitudes = np.maximum(
        0.4,
        rng.normal(cfg.fast_event_mean_mv, cfg.fast_event_sd_mv, size=len(event_times_s)),
    )
    for event_t, amplitude in zip(event_times_s, amplitudes):
        _add_events(voltage, np.asarray([event_t]), cfg.fs_hz, amplitude * kernel)


def _add_slow_depolarizations(
    voltage: np.ndarray,
    event_times_s: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> None:
    if len(event_times_s) == 0:
        return
    for event_t in event_times_s:
        duration_ms = max(50.0, rng.normal(cfg.slow_event_duration_ms, 45.0))
        amplitude = max(1.0, rng.normal(cfg.slow_event_mean_mv, cfg.slow_event_sd_mv))
        kernel = amplitude * _raised_cosine_plateau(cfg.fs_hz, duration_ms)
        _add_events(voltage, np.asarray([event_t]), cfg.fs_hz, kernel)


def _add_bursts(
    voltage: np.ndarray,
    burst_times_s: np.ndarray,
    duration_s: float,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(burst_times_s) == 0:
        return np.array([], dtype=float)

    ap_kernel = _ap_kernel(cfg)
    burst_ap_times = []
    for burst_t in burst_times_s:
        n_spikes = rng.integers(cfg.burst_spike_count_min, cfg.burst_spike_count_max + 1)
        interval = 1.0 / cfg.burst_frequency_hz
        jitter = rng.normal(0.0, interval * 0.08, size=n_spikes)
        spike_times = burst_t + np.arange(n_spikes) * interval + jitter
        spike_times = spike_times[(spike_times >= 0.0) & (spike_times < duration_s)]
        burst_ap_times.extend(spike_times.tolist())

        burst_duration_ms = max(45.0, (n_spikes + 2) * interval * 1000.0)
        plateau_amp = max(1.0, rng.normal(cfg.burst_plateau_mean_mv, cfg.burst_plateau_sd_mv))
        plateau = plateau_amp * _raised_cosine_plateau(cfg.fs_hz, burst_duration_ms)
        _add_events(voltage, np.asarray([burst_t]), cfg.fs_hz, plateau)
        _add_events(voltage, spike_times, cfg.fs_hz, ap_kernel)

    return np.sort(np.asarray(burst_ap_times, dtype=float))


def _add_hyperpolarizations(
    voltage: np.ndarray,
    event_times_s: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> None:
    if len(event_times_s) == 0:
        return
    kernel = _alpha_kernel(cfg.fs_hz, tau_rise_ms=2.0, tau_decay_ms=28.0)
    amplitudes = np.minimum(
        -0.4,
        rng.normal(cfg.ipsp_mean_mv, cfg.ipsp_sd_mv, size=len(event_times_s)),
    )
    for event_t, amplitude in zip(event_times_s, amplitudes):
        _add_events(voltage, np.asarray([event_t]), cfg.fs_hz, amplitude * kernel)


def _add_ripples(
    voltage: np.ndarray,
    ripple_times_s: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> None:
    if len(ripple_times_s) == 0:
        return

    dt = 1.0 / cfg.fs_hz
    duration_s = cfg.ripple_duration_ms / 1000.0
    t = np.arange(0.0, duration_s, dt)
    window = np.hanning(len(t))
    for event_t in ripple_times_s:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ripple = cfg.ripple_amplitude_mv * window * np.sin(
            2.0 * np.pi * cfg.ripple_frequency_hz * t + phase
        )
        _add_events(voltage, np.asarray([event_t]), cfg.fs_hz, ripple)


def _voltage_to_steady_dff(voltage_mv: np.ndarray, cfg: Jedi3SubConfig) -> np.ndarray:
    delta_mv = voltage_mv - cfg.resting_mv
    dff = np.zeros_like(delta_mv, dtype=float)

    depol = delta_mv >= 0
    depol_norm = np.tanh(delta_mv[depol] / cfg.depol_response_scale_mv)
    depol_den = np.tanh(cfg.depol_reference_mv / cfg.depol_response_scale_mv)
    dff[depol] = cfg.full_depol_dff * depol_norm / depol_den

    hyper = ~depol
    hyper_norm = np.tanh(np.abs(delta_mv[hyper]) / cfg.hyperpol_response_scale_mv)
    hyper_den = np.tanh(cfg.hyperpol_reference_mv / cfg.hyperpol_response_scale_mv)
    dff[hyper] = cfg.full_hyperpol_dff * hyper_norm / hyper_den
    return dff


def _delay_trace(signal: np.ndarray, delay_ms: float, fs_hz: float) -> np.ndarray:
    delay = int(round(delay_ms * fs_hz / 1000.0))
    if delay <= 0:
        return signal

    delayed = np.empty_like(signal)
    delayed[:delay] = signal[0]
    delayed[delay:] = signal[:-delay]
    return delayed


def _apply_jedi3sub_kinetics(dff_steady: np.ndarray, cfg: Jedi3SubConfig) -> np.ndarray:
    dt = 1.0 / cfg.fs_hz
    tau_fast = cfg.depol_tau_fast_ms / 1000.0
    tau_slow = cfg.depol_tau_slow_ms / 1000.0
    tau_repol = cfg.repol_tau_ms / 1000.0
    alpha_fast = 1.0 - np.exp(-dt / tau_fast)
    alpha_slow = 1.0 - np.exp(-dt / tau_slow)
    alpha_repol = 1.0 - np.exp(-dt / tau_repol)

    out = np.empty_like(dff_steady)
    fast_state = dff_steady[0]
    slow_state = dff_steady[0]
    out[0] = dff_steady[0]

    for i in range(1, len(dff_steady)):
        target = dff_steady[i]
        if target < out[i - 1]:
            fast_state += alpha_fast * (target - fast_state)
            slow_state += alpha_slow * (target - slow_state)
            out[i] = (
                cfg.depol_fast_fraction * fast_state
                + (1.0 - cfg.depol_fast_fraction) * slow_state
            )
        else:
            recovered = out[i - 1] + alpha_repol * (target - out[i - 1])
            fast_state = recovered
            slow_state = recovered
            out[i] = recovered

    return out


def _ap_artifact_trace(
    n_frame: int,
    ap_times_s: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    artifact = np.zeros(n_frame, dtype=float)
    if len(ap_times_s) == 0 or cfg.ap_artifact_amplitude == 0:
        return artifact

    sigma = max(0.5, cfg.ap_artifact_width_ms * cfg.fs_hz / 1000.0)
    radius = max(3, int(np.ceil(5.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    positive = np.exp(-0.5 * (x / sigma) ** 2)
    positive /= np.max(positive)

    after_sigma = max(1.0, 0.45 * cfg.fs_hz / 1000.0)
    after_x = np.arange(0, max(3, int(np.ceil(5.0 * after_sigma))) + 1)
    after = np.exp(-0.5 * (after_x / after_sigma) ** 2)
    after /= np.max(after)

    amplitudes = np.maximum(
        0.15,
        rng.normal(cfg.ap_artifact_amplitude, cfg.ap_artifact_sd, size=len(ap_times_s)),
    )
    for event_t, amplitude in zip(ap_times_s, amplitudes):
        center = int(round(event_t * cfg.fs_hz))
        start = max(0, center - radius)
        stop = min(n_frame, center + radius + 1)
        k_start = start - (center - radius)
        artifact[start:stop] += amplitude * positive[k_start : k_start + stop - start]

        after_start = center + 1
        after_stop = min(n_frame, after_start + len(after))
        if after_start < after_stop:
            artifact[after_start:after_stop] += (
                cfg.ap_artifact_afterpulse
                * amplitude
                * after[: after_stop - after_start]
            )
    return artifact


def _colored_noise(
    n_frame: int,
    alpha: float,
    sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sd <= 0:
        return np.zeros(n_frame, dtype=float)
    noise = np.empty(n_frame, dtype=float)
    noise[0] = rng.normal(0.0, sd)
    innovation_sd = sd * np.sqrt(max(1e-12, 1.0 - alpha**2))
    for i in range(1, n_frame):
        noise[i] = alpha * noise[i - 1] + rng.normal(0.0, innovation_sd)
    return noise


def _apply_linescan_observation_noise(
    dff_clean: np.ndarray,
    t: np.ndarray,
    cfg: Jedi3SubConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    photons_per_sample = cfg.photon_rate_hz / cfg.fs_hz
    bleach = 1.0 - cfg.bleach_fraction * (1.0 - np.exp(-t / cfg.bleach_tau_s))
    gain = 1.0 + _colored_noise(len(t), alpha=0.985, sd=cfg.scanline_gain_sd, rng=rng)
    motion = _colored_noise(len(t), alpha=0.9995, sd=cfg.motion_artifact_dff, rng=rng)

    fractional_signal = np.clip(1.0 + dff_clean + motion, 0.02, None)
    expected = photons_per_sample * bleach * gain * fractional_signal
    expected = np.clip(expected, 0.1, None)

    observed = rng.poisson(expected).astype(float)
    observed += rng.normal(0.0, cfg.read_noise_photons, size=len(dff_clean))
    dff = (observed - photons_per_sample) / photons_per_sample
    dff += rng.normal(0.0, cfg.electronic_noise_dff, size=len(dff_clean))
    return dff
