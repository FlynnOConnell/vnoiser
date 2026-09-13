# vnoiser

`vnoiser` denoises one-dimensional voltage-imaging traces, detects candidate
events, and provides an interactive notebook for manual event curation and
template matching.

The main user workflow is [notebooks/curation.ipynb](notebooks/curation.ipynb).
It supports manual curation, fast-event template matching, and independent
slow-event curation below 40 Hz.

## Quick Start

**Inputs**

- A Femtonics line-scan `.mesc` (each line-scan unit is one scan) opened through `mbo_utilities`, or the lab's packaged `VI_<date>.pkl`.
- `domains.json`: which lines make each domain, plus the scans in order.

```json
{"domains": {"soma1": [0, 1, 2], "basal1": [3, 4, 5]}, "scans": ["35", "38"], "first_env": ["35"]}
```

**Outputs** (`<animal>/<expt>/PF/`, what the curation dashboard reads)

| file | contents |
|---|---|
| `denoised_trace_scans.pkl` | `{scan: {domain: trace}}`, the curated trace |
| `fs_scans.pkl`, `scanIDs_ROIs.pkl` | frame rate, scan / domain / ROI tables |
| `detected_events_peaks.pkl`, `param_spike_detect.pkl` | peaks per domain, thresholds |
| `denoised_trace_components.pkl`, `test.h5` | masked sum, baselines, envelope; per-domain dF/F and z |
| `cwts.h5` (optional) | wavelet coefficients |
| `pipeline.json` | source, every parameter, versions |
| `.curation/*.json` | your Yes/No decisions |

**Parameters** (defaults reproduce the archive; `Denoiser.upstream`)

| stage | parameters |
|---|---|
| dF/F, z (`DfofConfig`) | baseline sigma 1500 samples, z baseline sigma 5000, first 1000 samples replaced, sign flipped |
| denoiser (`Denoiser`) | 100 log scales 1–1000, PCA 30/10, 5 clusters / 10 bands, open levels 2.0 / 2.5 SD, soft masks 0.7 / 0.5 / 0.2 / 0.01, 1 Hz FIR baseline |
| peaks (`SpikeDetectConfig`) | band-pass 2–400 Hz, 3.5 SD band-passed, 4 SD amplitude, 5 ms minimum duration, 3-sample spacing |

**GUI**

```bash
mbo stan112_expt12.mesc          # a .mesc with line scans opens the curation window
```

The Recordings tab lists every raw line; clicking one denoises that line alone
(about 1 min per 100 s of recording, cached after). For the domain traces:

1. Pipeline tab.
2. Check the output folder (defaults to `PF` beside the file).
3. Tick the scans; name the domains in the table (or `Load` a `domains.json`).
4. Pipeline Settings only to leave the defaults.
5. `Run Voltage`; the status line shows the worker's PID, the log is under `~/.mbo/logs`.
6. `Open in Curation` once the folder is written.

**CLI**

```bash
mbo voltage stan112_expt12.mesc --init           # writes domains.json next to the file; edit it
mbo voltage stan112_expt12.mesc                  # every scan in domains.json -> PF
mbo curate X:/data/asako/stan112/stan112_expt12  # curation window on that PF
```

**Python**

```python
from mbo_utilities.vnoiser.pipeline import run_voltage_pipeline
run_voltage_pipeline("stan112_expt12.mesc", domains={"soma1": [0, 1, 2], "basal1": [3, 4, 5]}, units=["MUnit_35"])

from vnoiser import run_from_vi, load_scan_rois   # from the packaged pickle instead
rois = load_scan_rois("PF/scanIDs_ROIs.pkl")
run_from_vi("VI_2025-07-24.pkl", "PF_new", domains=rois["domains"], scan_ids=["35"])
```

## Installation

From the repository root, create the supplied conda environment:

```bash
conda env create -f environment.yml
conda activate vnoiser
```

The environment installs `vnoiser` in editable mode with JupyterLab, Plotly,
and the test dependencies. To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate vnoiser
```

## Start Curation

1. Start the notebook:

   ```bash
   jupyter lab notebooks/curation.ipynb
   ```

2. In the first code cell, set `DATA_PATH` to the data location:

   ```python
   from pathlib import Path

   DATA_PATH = Path(
       "/biohpc/data4/asako/AOD/CA1_JEDI/spatial_JEDI3sub/Data"
   )
   ```

3. Run all notebook cells. This creates the three curation dashboards but does
   not load or process every trace.

4. For a shared spatial JEDI `Data/` path, select an **animal**, then one
   **experiment**. Only that experiment's small scan metadata is read. Select a
   scan/domain from the `recording` menu and click **Load**. A processed
   recording is displayed as:
   `animal / experiment / scan / domain`.

5. Review each candidate and click **Yes**, **No**, or **Clear**. Every decision
   is written to disk immediately.

6. Reopen the notebook and load the same recording to continue. Existing manual
   decisions are restored from its `.curation` file.

If the path is changed inside a dashboard rather than in the notebook cell,
click **Scan** before selecting a recording.

## The Three Curation Modes

All three sections can be used on the same recording in one notebook session.
They maintain separate candidates, templates, labels, and JSON files.

### 1. Manual

- Detects candidates by thresholding the denoised trace.
- Starts without a template or automatic classification.
- Each event marked **Yes** is added to the evolving template.
- Events marked **No** are excluded from the second-pass preview.
- **Clear** removes the manual decision.

Use this mode when the template should be built entirely from human-selected
events, including lower-amplitude events.

### 2. Fast

- Detects candidates by thresholding the denoised trace shown in black.
- Seeds the initial template from the top 25% highest-amplitude detected events.
- Compares every candidate with the template using cosine similarity.
- The default automatic pass threshold is cosine similarity greater than
  `0.80`.
- The amplitude threshold controls candidate inclusion; the cosine threshold
  separately controls the provisional pass/reject color.
- An adjustable auto-pass amplitude marks every candidate at or above it as a
  provisional pass regardless of cosine similarity.
- Cosine-based auto-rejection can be switched off, leaving candidates below the
  auto-pass amplitude unlabeled.
- Light green and light red are provisional automatic calls.
- A manual **Yes** or **No** overrides the automatic call and is shown with a
  darker color.
- **Clear** removes the manual override and returns the event to its automatic
  state.

Use this mode for fast, spike-like events when strong events provide a useful
initial template.

### 3. Slow

- Applies a zero-phase `<40 Hz` low-pass filter directly to the denoised trace.
- Detects candidates by thresholding that filtered trace, shown in blue.
- Seeds its own template from the top 25% highest-amplitude slow candidates.
- Uses the same cosine-similarity, auto-pass amplitude, waveform-rejection
  toggle, and manual-override behavior as Fast mode.
- Uses a wider `-500 ms` to `+500 ms` template window.
- Never changes the Fast mode template or labels.

Use this mode for slower, subthreshold-like events that should not be forced to
match a fast spike template.

## Reading The Dashboard

- **A. Full trace and candidate events:** shows the complete z-scored trace.
  The denoised trace is black. Slow mode also overlays the direct `<40 Hz`
  analysis trace in blue. Click a candidate marker to focus that event. The
  outlined marker and vertical line identify the current event.
- **A1. Threshold:** moves the horizontal candidate threshold. Fast and Manual
  apply it to the black denoised trace; Slow applies it to the blue filtered
  trace. Its initial value and useful range are derived from the selected
  trace's robust noise and peak levels. Candidate detection, the top-25% seed
  template, cosine scores, and all downstream panels update after the slider is
  released. The selected threshold is saved separately for each recording and
  mode.
- **A2. Auto-pass (Fast and Slow only):** the slider sets an amplitude at or
  above which every candidate is auto-called pass, drawn as a dotted green line
  in panel A. It starts at the top of the range, so nothing is auto-passed
  until it is lowered. The **waveform reject** checkbox turns cosine-based
  auto-rejection on or off; when off, candidates below the auto-pass amplitude
  and cosine threshold stay unlabeled instead of light red. Both settings are
  saved separately for each recording and mode.
- **B. Current template:** shows all events contributing to the template at low
  opacity and their average as the dark template line.
- **C. Focused candidate:** compares the selected event with the current
  template and reports cosine similarity. Fast events default to a
  `-100 ms` to `+100 ms` view; Slow mode uses `-500 ms` to `+500 ms`.
- **D. Second pass preview:** previews the denoised trace after all manually
  rejected events are suppressed. Rejected regions are replaced by local linear
  interpolation. This is a preview and does not overwrite the source trace.
- **E. Candidate PCA (400 ms):** projects the centered `-200 ms` to `+200 ms`
  window around every candidate into PC1/PC2 space. Points retain the current
  automatic or manual decision colors. Click a point to focus that event.
- **Decision:** **Yes**, **No**, and **Clear** update the focused event and save
  immediately. Manually labeled events remain available when a threshold change
  would otherwise remove them; use **Clear** to release that retained event.
- **Navigation:** use the previous/next buttons or event slider to move through
  candidates.
- **View:** filter the dashboard to All, Yes, No, or Unlabeled manual decisions.

## Saved Curation Data

For processed spatial JEDI data, labels are stored inside the selected
experiment's `PF/.curation/` directory:

```text
PF/
├── denoised_trace_scans.pkl
├── fs_scans.pkl
├── scanIDs_ROIs.pkl
└── .curation/
    ├── manual_template_curation.json
    ├── fast_template_curation.json
    └── slow_template_curation.json
```

Each manually curated event stores its recording identifier, source sample and
time, aligned sample, amplitude, current template similarity, initial automatic
call, manual label, and update time. Each mode file also stores the selected
candidate threshold, auto-pass amplitude, and waveform-rejection state
separately for every recording.

The dashboard never modifies the source pickle files. To transfer curation with
a dataset, include the experiment's `.curation/` directory.

For raw MATLAB recordings, `.curation/` is created beside the selected file or
inside the selected recording directory.

## Supported Data Paths

`DATA_PATH` may point to any of the following:

- The shared spatial JEDI `Data/` directory.
- One animal directory.
- One experiment directory.
- One experiment's `PF/` directory.
- A specific `denoised_trace_scans.pkl` file.
- A directory of supported raw `.mat` recordings.
- One supported raw `.mat` recording.

Processed spatial JEDI data is expected to follow this structure:

```text
Data/
└── <animal>/
    └── <experiment>/
        └── PF/
            ├── denoised_trace_scans.pkl
            ├── fs_scans.pkl
            └── scanIDs_ROIs.pkl
```

Candidate events are derived from the selected trace at load time. Existing
event catalogs such as `detected_events_peaks.pkl` are not used to seed or veto
Manual, Fast, or Slow candidates.

At startup the loader lists only the immediate animal directories. Selecting an
animal lists only its immediate experiment directories. Scan/ROI metadata is
opened only after one experiment is selected, and the large trace pickle is
opened only after **Load** is clicked. Dashboard sections share cached metadata
and the in-memory trace within the same Python session. Processed traces are
used directly and do not rerun the wavelet denoiser.

Raw `.mat` files must contain a `CAttached` structure with `fluo_mean` and
`fluo_time`. `events_AP` is optional.

## Denoising API

The denoiser accepts one z-scored dF/F trace:

```python
import numpy as np

from vnoiser import Denoiser

dfof_z = np.asarray(..., dtype=float)
model = Denoiser(fs=1000.0)
denoised_trace, event_indices = model.run(dfof_z)
```

The pipeline computes a continuous wavelet transform, clusters frequency bands
with PCA and hierarchical clustering, detects event windows, masks unlikely
activity, reconstructs event-related signal, and restores the slow baseline.

## Notebooks

- [curation.ipynb](notebooks/curation.ipynb): the user-facing curation workflow.
- [pipeline.ipynb](notebooks/pipeline.ipynb): interactive walkthrough of the
  denoising and template-matching pipeline.
- [jedi3sub_simulation.ipynb](notebooks/jedi3sub_simulation.ipynb): JEDI3sub
  simulator exploration and calibration.
- [wavelet_denoising_walkthrough.ipynb](notebooks/wavelet_denoising_walkthrough.ipynb):
  stage-by-stage review of the upstream wavelet denoising and event
  detection on one processed scan/domain, compared with the saved `PF/`
  files, and of what `curation.ipynb` computes at load time.

## Repository Layout

```text
.
├── vnoiser/       # Installable package
├── notebooks/     # Interactive workflows
├── tests/         # Automated tests
├── docs/          # Reference parity notes
├── ref/           # Original source used for cross-checking
├── pyproject.toml
└── environment.yml
```

Run the test suite with:

```bash
pytest -q
```

The original source comparison is documented in
[docs/ref_parity_audit.md](docs/ref_parity_audit.md).

## Overview

### Layout

| Path | Purpose | Lines |
| --- | --- | --- |
| `vnoiser/` | Installable package | ~3,800 |
| `notebooks/` | Curation, pipeline walkthrough, denoising walkthrough, simulator notebooks | 4 notebooks |
| `tests/` | pytest suite, 31 test functions | ~1,000 |
| `ref/` | Original source kept only for cross-checking | ~460 |
| `docs/ref_parity_audit.md` | What was ported from `ref/` and what deliberately changed | |
| `pyproject.toml`, `environment.yml` | setuptools build, conda env (Python 3.11) | |

### Package modules

| Module | Key symbols | Role |
| --- | --- | --- |
| `denoiser.py` | `Denoiser`, `FrequencyClusterer`, `WaveletReducer`, `AdaptiveThreshold`, `trigger_detect`, 3 config dataclasses | CWT → PCA + hierarchical clustering of frequency bands → event windows → masking → reconstruction → FIR baseline restore |
| `dataset.py` | `SpatialJediDataset`, `JediSub3Dataset`, `open_recording_dataset`, `SpatialTraceRef`, `RecordingSample` | Lazy loaders for processed spatial JEDI pickles (`Data/<animal>/<experiment>/PF/`) and raw `.mat` files; restricted unpickler; `lru_cache` on file size and mtime |
| `curation.py` | `EventCurationDashboard`, `CandidateSet`, threshold and PCA helpers | ipywidgets + Plotly dashboard with three modes; writes JSON labels to `PF/.curation/` on every click |
| `simulation.py` | `simulate_jedi3sub_trace`, `Jedi3SubConfig`, `SimulationResult` | Synthetic JEDI3sub linescan traces: APs, slow drift, bursts, ripples, indicator kinetics, colored noise |

### Curation modes

| Mode | Candidate source | Template seed | Auto call | Window |
| --- | --- | --- | --- | --- |
| Manual | Threshold on denoised trace | Built only from Yes events | None | ±100 ms |
| Fast | Threshold on denoised trace | Top 25% amplitude | Amplitude ≥ auto-pass, else cosine > 0.80 (toggleable) | ±100 ms |
| Slow | Threshold on <40 Hz low-passed trace | Top 25% amplitude | Amplitude ≥ auto-pass, else cosine > 0.80 (toggleable) | ±500 ms |

### Dependencies

- Core: numpy, scipy, scikit-learn, PyWavelets
- Notebook extra: plotly, ipywidgets, anywidget, jupyterlab, nbformat, ipykernel
- Dev / reference extras: pytest, h5py

### Tests

| File | Count | Covers |
| --- | --- | --- |
| `test_curation.py` | 23 | Lazy loading, JSON save and reload, per-mode file isolation, threshold slider scaling, PCA window, label retention, notebook cell contents |
| `test_dataset.py` | 4 | `.mat` windowing, lazy spatial metadata, format auto-detection |
| `test_simulation.py` | 4 | Shape, sign of deflections, seed determinism, denoiser on short trace |

### Notes

- Primary entry point is `notebooks/curation.ipynb`; set `DATA_PATH` in the first cell.
- Source pickles are never modified. Curation state lives entirely in `.curation/*.json`, one file per mode, thresholds stored per recording.
- Processed traces skip the denoiser. The wavelet pipeline runs only on raw `.mat` input or via the API directly.
- `ref/` is not installed. Parity differences versus the port are documented, including changed soft-threshold attenuation values and a fixed config typo.
- A stale `vnoiser.egg-info/` sits at repo root from an editable install.

## Auto-Pass and Waveform-Rejection Controls

Added after a collaborator reported that too many large-amplitude Slow events
were auto-rejected by the seed-template cosine check.

- `EventCurationDashboard` accepts `auto_pass_amplitude` and
  `waveform_rejection` and exposes both in panel A2 for Fast and Slow modes.
- `_initial_auto_call_for_index` passes any candidate at or above the auto-pass
  amplitude, then applies the `0.80` cosine rule, and returns no call instead
  of reject when waveform rejection is off.
- Manual **Yes** and **No** labels still override every automatic call.
- Settings are stored under `candidate_detection.auto_pass_amplitudes` and
  `candidate_detection.waveform_rejection`, keyed by recording, and echoed on
  each saved event.
- The collaborator's copy lives on the cluster at
  `/biohpc/data4/asako/AOD/CA1_JEDI/spatial_JEDI3sub/Notebooks/vnoiser/`. This
  directory is not a git repository, so `vnoiser/curation.py`,
  `notebooks/curation.ipynb`, and `tests/test_curation.py` must be copied there
  by hand.
