from dataclasses import dataclass, field
from typing import List

@dataclass
class ClusteringConfig:
    n_components: int = 30
    n_comp_clu: int = 10
    n_clusters: int = 5
    n_subclusters: int = 10
    thres_dist: float = 0.4
    standardize: bool = True
    method_linkage: str = "ward"
    method_fclust: str = "maxclust"

@dataclass
class cwtReducerConfig:
    cluster_label: str = "sublabels"
    cluster_freq: str = "subfreqs"
    # reduce_method: srt = "average"
    cutoff_freq: tuple = (0, 400)
    slow_upthres: float = 2.0
    fast_upthres: float = 2.5

@dataclass
class thresConfig:
    thres_type: str = "hard"

@dataclass
class DfofConfig:
    exclude_val: List[str] = field(default_factory=lambda: ['All_domains','bg'])
    negative: bool = True
    filename: str = "dfof_zscore.h5"
    filepath: str = "./"