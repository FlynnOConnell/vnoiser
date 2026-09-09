# Notebooks

Run notebooks from the repository's `vnoiser` conda environment. Complete
curation instructions are in the [main README](../README.md#start-curation).

- `curation.ipynb`: user-facing manual, Fast, and Slow event curation. This is
  the notebook to use on experimental data.
- `pipeline.ipynb`: technical walkthrough of denoising and template matching.
- `jedi3sub_simulation.ipynb`: simulator calibration and exploration.

For processed spatial JEDI data, curation is saved beside the selected
experiment under `PF/.curation/`. Source traces are never overwritten.
