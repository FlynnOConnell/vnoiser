import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from .config import ClusteringConfig

class FrequencyClusterer:
    def __init__(self, coeff, freqs, config: ClusteringConfig = None):
        self.coeff = coeff
        self.freqs = freqs

        # use provide config or defaults
        self.config = config or ClusetringConfig()


    def run(self):
        # Normalize
        if self.config.standardize:
            mean = np.mean(self.coeff, axis=1, keepdims=True)
            std = np.std(self.coeff, axis=1, ddof=0, keepdims=True)
            cwt_coeff = (self.coeff - mean) / std
        else:
            cwt_coeff = self.coeff

        # PCA
        pca = PCA(n_components=self.config.n_components)
        pca_coeff = pca.fit_transform(np.abs(cwt_coeff))
        explained = pca.explained_variance_ratio_

        # Clustering
        Z = linkage(pca_coeff[:,:self.config.n_comp_clu], method=self.config.method_linkage)
        if self.config.method_fclust == 'maxclust':
            labels = fcluster(Z, self.config.n_clusters, criterion='maxclust')
            sublabels = fcluster(Z, self.config.n_subclusters, criterion='maxclust')
        elif self.config.method_fclust == 'distance':
            threshold = self.config.thres_dist * max(Z[:,2])
            labels = fcluster(Z, t=threshold, criterion='distance')
            sublabels = fcluster(Z, t=threshold/2, criterion='distance')
        else:
            raise ValueError(f"Unsupported method_fclust: {self.config.method_fclust}")

        return {
            "pca_coeff": pca_coeff,
            "explained": explained,
            "Z": Z,
            "labels": labels,
            "frequencies": {i: self.freqs[labels == i] for i in range(1, self.config.n_clusters+1)},
            "sublabels": sublabels,
            "subfreqs": {i: self.freqs[sublabels == i] for i in range(1, self.config.n_subclusters+1)}
        }
