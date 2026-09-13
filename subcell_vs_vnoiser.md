# subcell vs vnoiser

Comparison of two sibling repositories under `~/repos`, written 2026-09-10.

**Short answer:** `vnoiser` is the wavelet denoising and event detection code.
It is a Python port of a reference implementation kept under `ref/`, whose
helper package is named `utils_st/` (the `st` initials, PSTH helpers, and the
`calc_dfof_gauss` / `trigger_detect` utilities are the original author's
signal-utility module). Neither repository names Satoshi Terada in a file,
docstring, or commit, so the attribution rests on that `utils_st` naming, the
Losonczy Lab author field in `pyproject.toml`, and the JEDI voltage-imaging
data paths on the cluster. `subcell` is unrelated to that lineage: it ports a
MATLAB two-photon *Bergamo* pipeline (`stripRegistrationBergamo.m`,
`summarize_LoCo.m`, `localizeSources_vIM.m`) for synaptic glutamate imaging.

## At a glance

| | `subcell` | `vnoiser` |
| --- | --- | --- |
| One-line description | Synaptic signal extraction from two-photon Bergamo imaging data | Wavelet-based denoising and event detection for single z-scored dF/F traces |
| Scientific domain | 2-photon glutamate imaging of dendritic spines / synapses, multi-trial movies | AOD voltage imaging (JEDI, JEDI3sub), 1-D linescan traces at ~1 kHz |
| Input data | Multi-page ScanImage `.tif` movies (one per trial), optional mbo LazyArray | Processed spatial JEDI pickles (`PF/denoised_trace_scans.pkl`, `fs_scans.pkl`, `scanIDs_ROIs.pkl`) or raw `.mat` with `CAttached.fluo_mean` / `fluo_time` |
| Data dimensionality | 3-D + trials (y, x, t, trial, channel) | 1-D time series (one trace at a time) |
| Origin being ported | MATLAB: `stripRegistrationBergamo.m`, `summarize_LoCo.m`, `localizeSources_vIM.m`, `extractTrial.m`, `dftregistration_clipped.m` | Python reference package in `ref/` (`clustering.py`, `reducer.py`, `thresholding.py`, `reconstructor.py`, `utils_st/signal_utils.py`) |
| Wavelet denoising | No | Yes, the core of the package (CWT with `cmor0.5-1.0`) |
| Event detection | Indirect: NMF temporal components, then event/denoised/F0/SNR variants in `assemble.py` | Yes, explicit: threshold crossings via `trigger_detect`, per-cluster event windows, plus curation-time threshold candidates |
| Motion correction | Yes, DFT sub-pixel registration with template building and quality metrics | No |
| Source / ROI localization | Yes (matched filter, DoG, NMS, activity image, density threshold, sub-pixel refinement) | No, ROIs come pre-defined from the pickles |
| Signal unmixing | Constrained NMF per connected component, PyTorch L-BFGS replacing MATLAB `fmincon` | No |
| Manual curation UI | No (viewers only) | Yes, three-mode ipywidgets + Plotly dashboard with template matching |
| Author field | none (Flynn O'Connell commits) | `Losonczy Lab`, license `Proprietary` |
| Package size | ~11,800 lines across 45 modules | ~4,000 lines across 4 modules |
| Tests | 18 test functions in one file (`test_subcell_vis.py`) plus a synthetic movie fixture; `slow` marker for real registration + NMF | 31 test functions across `test_curation.py` (23), `test_dataset.py` (4), `test_simulation.py` (4) |

## Pipeline stages

| Stage | `subcell` | `vnoiser` |
| --- | --- | --- |
| 1. Load | `io/tiff_reader.py`, `io/lazy_array.py`, `io/trial_table.py` build a trial table from TIFFs or an mbo LazyArray | `dataset.py`: `SpatialJediDataset` (lazy animal, experiment, scan/domain), `JediSub3Dataset` (raw `.mat`), restricted unpickler, `lru_cache` keyed on size and mtime |
| 2. Preprocess | `registration/`: downsample in time, build template, DFT register each strip, upsample motion, score alignment quality; writes registered movie + `AlignmentData` to Zarr | z-scored dF/F expected as input; `ref/preprocessor.py` (dF/F, z-score) is kept for reference only and is not installed |
| 3. Core transform | `extraction/localize.py`: moving mean, moving median baseline, MAD noise, z-score, exponential matched filter, DoG, spatiotemporal NMS, activity image, peak + density threshold, sub-pixel refinement | `denoiser.py`: CWT, then `FrequencyClusterer` (PCA + Ward hierarchical clustering of frequency bands), then `WaveletReducer` (per-cluster reduced coefficient trace, Savitzky-Golay smoothing, threshold-crossing event windows), then `AdaptiveThreshold` (soft/hard mask), then reconstruction, then zero-phase FIR low-pass baseline restore |
| 4. Cross-trial / cross-event | `cross_trial_align.py` aligns activity images across trials, drops trials below a correlation floor; `source_selection.py` picks sources from the averaged activity image | Template matching in `curation.py`: seed template from top 25 % amplitude candidates, cosine similarity > 0.80, auto-pass amplitude, waveform-rejection toggle, candidate PCA embedding |
| 5. Extraction / output | `extract_trial.py` + `nmf/` (baseline, objective, spatial, solver) per trial; `assemble.py` produces events, denoised, least-squares, F0, SNR; results land in a Zarr `ExperimentStore` | `Denoiser.run(dfof_z)` returns `(denoised_trace, event_indices)`; curation writes `PF/.curation/{manual,fast,slow}_template_curation.json`, never touches source pickles |
| 6. Parallelism / compute | Multiprocess registration workers, parallel extraction workers, torch device auto/cpu/cuda | Single process, NumPy / SciPy / scikit-learn / PyWavelets only |
| Entry points | CLI `subcell` with `init-config`, `build-trial-table`, `register`, `extract`, `run`, `vis`; mbo_utilities pipelines entry point | No CLI; `notebooks/curation.ipynb` is the primary entry point, `Denoiser` is the API |

## Visualization and interactivity

| | `subcell` | `vnoiser` |
| --- | --- | --- |
| Stack | Panel + Bokeh (`ExtractionViewer`), fastplotlib + imgui (`SubcellVis` NDWidget) | ipywidgets + Plotly + anywidget inside JupyterLab |
| What is shown | Field of view with trial/time sliders, raw and registered movies, mean/activity images, NMF footprints, source centers, traces, motion, trial table, pipeline log | Full z-scored trace with candidate markers, current template, focused candidate vs template, second-pass preview, candidate PCA, decision buttons |
| Persistence of user actions | Labels/filters in the viewer session | Every Yes/No/Clear and every slider written to JSON immediately, restored on reload |

## Repository layout

| Path | `subcell` | `vnoiser` |
| --- | --- | --- |
| Package | `subcell/` with `registration/`, `extraction/` (+ `nmf/`), `filters/`, `io/`, `pipeline/`, `visualization/`, `_utils/`, `cli.py`, `config.py`, `mbo.py` | `vnoiser/` with `denoiser.py`, `dataset.py`, `curation.py`, `simulation.py` |
| Config | Pydantic `PipelineConfig` / `RegistrationConfig` / `ExtractionConfig`, YAML templates in `configs/defaults/` | Three dataclasses (`ClusteringConfig`, `cwtReducerConfig`, `thresConfig`) carried verbatim from `ref/config.py` |
| Docs | `docs/matlab_python_comparison.md` (66 kB step-by-step MATLAB vs Python audit) | `docs/ref_parity_audit.md` (what was ported from `ref/`, what deliberately changed) |
| Reference / validation code | `equivalency_tests/` with MATLAB scripts and Python comparison drivers | `ref/` original Python source, not installed |
| Notebooks | `examples/01` to `05` (motion correction, summarize, extraction viewer, MATLAB comparison, full pipeline), `subcell_capability_tour.ipynb`, `subcell_vs_suite2p.ipynb` | `notebooks/curation.ipynb`, `pipeline.ipynb`, `jedi3sub_simulation.ipynb`, `wavelet_denoising_walkthrough.ipynb` |
| Environment | `pyproject.toml` (setuptools), extras `viewer`, `examples`, `mbo`, `vis`, `dev`; ruff configured | `pyproject.toml` (setuptools), extras `reference`, `dev`, `notebook`; `environment.yml` conda env; `uv.lock` |
| Stray artifacts | `lightning_logs/` checkpoint, `.ruff_cache/`, `.ipynb_checkpoints/` | stale `vnoiser.egg-info/` noted in README |

## Dependencies

| | `subcell` | `vnoiser` |
| --- | --- | --- |
| Core | numpy, torch, scipy, pandas, bottleneck, tifffile, zarr>=3, numcodecs, pydantic, pyyaml, click, psutil | numpy, scipy, scikit-learn, PyWavelets |
| Optional | panel, bokeh, h5py, matplotlib, jupyter, mbo-utilities, fastplotlib, imgui-bundle, masknmf | plotly, nbformat, ipywidgets, anywidget, ipykernel, jupyterlab, h5py |
| Dev | pytest, pytest-cov, ruff | pytest |

## Where they could connect

The two packages do not import each other and share no code. The natural
seam is trace-level: `subcell` ends with per-source temporal signals stored in
Zarr, and `vnoiser.Denoiser.run` accepts any single z-scored trace at a given
`fs`. Running `vnoiser` on `subcell` output would require z-scoring the NMF
temporal components and confirming that the CWT frequency scales, which
assume ~1 kHz voltage imaging, are sensible at two-photon frame rates.

## Deviations from the reference implementations

| | `subcell` (vs MATLAB) | `vnoiser` (vs `ref/`) |
| --- | --- | --- |
| Optimizer | PyTorch L-BFGS with projection replaces `fmincon` trust-region-reflective | n/a |
| Numerics | Documented exact vs approximate steps per stage in `docs/matlab_python_comparison.md`; `movmad`, `ordfilt2`, `imgaussfilt` re-implemented | Soft-threshold attenuation 0.7 / 0.5 instead of 0.5 / 0.4; hard mode starts from 0.05 baseline mask; sigma 1 above 120 Hz in hard mode; low-frequency clusters rescaled to pre-smoothing peak; FIR baseline restore implemented in `Denoiser` |
| Bug fixes | Pipeline bugs fixed during the rename to `subcell` (commit 6c35260) | `ClusetringConfig` typo fixed; `reconstructor.py` left as reference because it references missing symbols |
| Storage | Zarr store instead of TIFF + `_ALIGNMENTDATA.mat` | JSON curation files instead of overwriting pickles |
