"""Fluorescence to z-scored dF/F, the way the spatial JEDI pipeline did it.

Stages 0-1 of the archive (``ref/preprocessor.py``, ``GetDfof``). A domain
(the soma, one dendritic branch) is several line ROIs; its fluorescence is
the mean over every pixel of those ROIs per frame, so an ROI weighs by its
pixel count. dF/F uses a Gaussian baseline; the z-score removes a slower
Gaussian baseline, divides by the SD and flips the sign, because JEDI darkens
on depolarisation. The first samples (the filter start-up) are replaced by
the mean of the rest before the z-score, in the dF/F that gets saved too.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

EXCLUDED_DOMAINS = ("All_domains", "bg")


@dataclass
class DfofConfig:
    """Parameters of :func:`domain_zscore`, in samples.

    The archive used these values at 1075 Hz: a 1.4 s dF/F baseline, a 4.7 s
    baseline under the z-score, and the first 0.93 s replaced.
    """

    sigma_dfof: float = 1500.0
    sigma_baseline: float = 5000.0
    n_startup: int = 1000
    negative: bool = True


@dataclass
class DomainTraces:
    """Per-domain dF/F and z-score, rows in ``names`` order."""

    names: list
    dfof_raw: np.ndarray
    z: np.ndarray


def domain_names(domains: Mapping[str, Sequence[int]], exclude=EXCLUDED_DOMAINS) -> list:
    """The domains that get a trace, in the mapping's order."""
    return [name for name in domains if name not in exclude]


def domain_zscore(
    traces,
    domains: Mapping[str, Sequence[int]],
    weights=None,
    cfg: DfofConfig | None = None,
    exclude=EXCLUDED_DOMAINS,
) -> DomainTraces:
    """Stages 0-1 for every domain of one scan.

    Parameters
    ----------
    traces : mapping of roi -> array (n_frame,)
        Mean fluorescence of each ROI per frame.
    domains : mapping of domain name -> ROIs (``domain_ROInumber``)
    weights : mapping of roi -> pixel count, optional
        Equal weights when omitted, which is the pixel mean only if every
        ROI of a domain has the same size.
    cfg : DfofConfig, optional
    exclude : names in ``domains`` that are not real domains

    Returns
    -------
    DomainTraces
        ``dfof_raw`` with the first ``cfg.n_startup`` samples of every row
        replaced by the mean of the rest (what the archive saved), and
        ``z = -(dfof - baseline) / SD(dfof)`` per row (``+`` when
        ``cfg.negative`` is off).
    """
    cfg = cfg or DfofConfig()
    names = domain_names(domains, exclude)
    if not names:
        raise ValueError("no domains to process")
    rows = []
    for name in names:
        rois = [int(r) for r in domains[name]]
        if not rois:
            raise ValueError(f"domain {name} has no ROIs")
        stack = np.stack([np.asarray(traces[r], dtype=float).ravel() for r in rois])
        if weights is None:
            rows.append(stack.mean(axis=0))
            continue
        w = np.asarray([float(weights[r]) for r in rois])
        if not np.all(w > 0):
            raise ValueError("ROI weights must be positive")
        rows.append((w[:, None] * stack).sum(axis=0) / w.sum())
    f = np.stack(rows)
    baseline = gaussian_filter1d(f, sigma=cfg.sigma_dfof, axis=1)
    dfof = (f - baseline) / baseline
    n = int(cfg.n_startup)
    if n > 0 and dfof.shape[1] > n:
        dfof[:, :n] = dfof[:, n:].mean(axis=1, keepdims=True)
    baseline = gaussian_filter1d(dfof, sigma=cfg.sigma_baseline, axis=1)
    z = (dfof - baseline) / dfof.std(axis=1, keepdims=True)
    if cfg.negative:
        z = -z
    return DomainTraces(names=names, dfof_raw=dfof, z=z)
