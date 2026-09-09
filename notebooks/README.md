# Notebooks

Run notebooks from the repository's `vnoiser` conda environment. Complete
curation instructions are in the [main README](../README.md#start-curation).

- `curation.ipynb`: user-facing manual, Fast, and Slow event curation. This is
  the notebook to use on experimental data.
- `pipeline.ipynb`: technical walkthrough of denoising and template matching.
- `jedi3sub_simulation.ipynb`: simulator calibration and exploration.
- `wavelet_denoising_walkthrough.ipynb`: stage-by-stage review of the
  upstream wavelet denoising and event detection on one processed
  scan/domain, compared with the saved `PF/` files. Expects `../data`.

For processed spatial JEDI data, curation is saved beside the selected
experiment under `PF/.curation/`. Source traces are never overwritten.
