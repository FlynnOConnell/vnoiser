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
- `ref/preprocessor.py`: dF/F extraction, z-scoring, and HDF5 writing.
- `ref/reconstructor.py`: rough reconstruction helper that references missing
  symbols and older result keys.
- `ref/utils_st/signal_utils.py`: PSTH helpers and `calc_dfof_gauss`, except
  for `trigger_detect`, which was carried into `vnoiser`.

The current package is intentionally focused on one already z-scored dF/F trace
at a time. Data loading, preprocessing, simulation, notebooks, and future
template-matching or ML tuning workflows can be added as separate modules under
`vnoiser`.
