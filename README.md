# vnoiser

`vnoiser` denoises one-dimensional voltage-imaging traces, detects candidate
events, and provides an interactive notebook for manual event curation and
template matching.

The main user workflow is [notebooks/curation.ipynb](notebooks/curation.ipynb).
It supports manual curation, fast-event template matching, and independent
slow-event curation below 40 Hz.

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
- Uses the same cosine-similarity and manual-override behavior as Fast mode.
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
candidate threshold separately for every recording.

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
