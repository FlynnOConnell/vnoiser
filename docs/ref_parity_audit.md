# Reference Parity Audit

This audit compares the installable `vnoiser` package with the original
reference code kept under `ref/`.

## Carried Over Into `vnoiser`

- Core denoising configuration: `ClusteringConfig`, `cwtReducerConfig`, and
  `thresConfig`.
- Threshold-crossing event detection via `trigger_detect`.
- Frequency clustering with PCA plus hierarchical clustering.
- Cluster-level wavelet reduction and event-window detection.
- Adaptive event masking and wavelet component reconstruction.
- End-to-end single-trace CWT wrapper.
- FIR low-pass baseline restoration in the final denoised trace.

## Intentional Differences From `ref/`

- `FrequencyClusterer` now uses `ClusteringConfig()` as its default config,
  fixing the original `ClusetringConfig` typo path.
- `WaveletReducer` rescales smoothed low-frequency cluster traces back toward
  their pre-smoothing peak amplitude.
- Soft-threshold attenuation starts at `0.7` for very low frequencies and
  `0.5` for 5-30 Hz clusters, instead of the older `0.5` and `0.4` values.
- Hard-threshold mode starts from a small `0.05` baseline mask rather than an
  all-zero mask.
- Hard-threshold smoothing uses sigma `1` above 120 Hz; soft thresholding keeps
  sigma `3` above 120 Hz.
- Final baseline restoration is implemented directly with a zero-phase FIR
  low-pass filter in `Denoiser`.

## Left As Reference-Only Context

- `ref/datamanager.py`: pickle/HDF5-specific loading helpers.
- `ref/reconstructor.py`: rough reconstruction helper that references missing
  symbols and older result keys.
- `ref/utils_st/signal_utils.py`: PSTH helpers, except for `trigger_detect`
  and `calc_dfof_gauss`, which live in `vnoiser` (`denoiser.py`,
  `preprocess.py`).

## Parity With The Archive (2026-09-12)

Checked against `stan112/stan112_expt12` (scans 35 and 38, eight domains,
1075.27 Hz), starting from the lab's packaged `VI_2025-07-24.pkl` and the
copy of the denoiser it was run with
(`analysis_notebooks/denoising/dendrowave/`, which differs from `ref/`).

| Stage | Code | Result |
|---|---|---|
| 0-1 pixels -> dF/F -> z | `preprocess.domain_zscore` | exact (`test.h5`, 1e-13). ROIs weigh by pixel count: three lines per scan are 20 px wide |
| 2 wavelet transform | `Denoiser._cwt` | exact to float32 (`cwts.h5`) |
| 3-5 bands, windows, masks | `Denoiser.upstream()` | exact (`rescaled_signal`, 1e-7) on all 16 traces with **soft** masks, the lab's attenuation above 80 Hz (`0.01`), complex band traces above 75 Hz |
| 6 baselines | `fir_lowpass(..., odd_taps=False)` | exact (1e-15) with the exact frame rate and the even tap count `int(fs * 2)` |
| final trace | `pf.final_trace` | `real(rescaled_signal) + lp_FIR1Hz[0]`: a per-trace constant the archive's `denoised_trace_scans.pkl` carries; a handful of samples per trace were patched by hand upstream |
| peaks | `events.detect_peaks` | exact on 7 of 16 traces with a 3-sample peak spacing; on the rest every archived peak is found plus one to nine extra events near the amplitude threshold |

What the port keeps as its own defaults (`Denoiser(fs)`): soft attenuation
`0.1` above 80 Hz, hard floor `0.05`, real band traces, odd FIR taps. They are
`thresConfig` / `cwtReducerConfig` / `Denoiser` fields now, and
`Denoiser.upstream(fs)` sets the archive's values.
