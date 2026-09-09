import numpy as np
from scipy.ndimage import gaussian_filter1d
from .config import thresConfig

class AdaptiveThreshold:
    def __init__(self, reduced_cwt, config: thresConfig = None):
        self.reduced = reduced_cwt
        self.n_clu, self.n_frame = np.shape(self.reduced["reduced_coeff"])

        self.config = config or thresConfig()

    def run(self):
        event_onsets = self.reduced["event_onsets"].copy()
        #selected_clusters = self.reduced['selected_clusters']
        selected_freqs = self.reduced['reduced_freqs'][:, 1].copy()

        num_labels = len(event_onsets)
        label_matrix = np.zeros((num_labels, self.n_frame))

        if self.config.thres_type == "soft":
            # Initial attenuation level by frequency
            for i, freq in enumerate(selected_freqs):
                if freq < 5:
                    label_matrix[i, :] = 0.5
                elif 5 <= freq < 30:
                    label_matrix[i, :] = 0.4
                elif 30 <= freq < 80:
                    label_matrix[i, :] = 0.2
                else:
                    label_matrix[i, :] = 0.1
        elif self.config.thres_type == "hard":
            label_matrix = label_matrix

        # Fill in detected event regions
        for i, (event_id, data) in enumerate(event_onsets.items()):
            crossings = data["crossings"]
            median_iei = np.median(data["iei"])

            if selected_freqs[i] <= 20:
                extra_bin = int(median_iei * 2.0)
            elif 20 < selected_freqs[i] <= 50:
                extra_bin = int(median_iei * 1.5)
            else:
                extra_bin = int(median_iei * 0.5)

            for st, ed in crossings:
                st_bin = max(st - extra_bin, 0)
                ed_bin = min(ed + extra_bin, self.n_frame)
                label_matrix[i, st_bin:ed_bin] = 1

        # Scale and smooth
        recon_data = self.reduced["reduced_coeff"][:, :].copy()
        scale_factors = np.sqrt(self.reduced["reduced_scale"][:, np.newaxis])
        recon_scaled = recon_data / scale_factors

        for i, freq in enumerate(selected_freqs):
            sigma = 10 if freq <= 120 else 3
            smooth_label = gaussian_filter1d(label_matrix[i], sigma=sigma)
            recon_data[i, :] *= smooth_label
            recon_scaled[i, :] *= smooth_label

        return {
            "clu_label": label_matrix,
            "all_scaled": recon_scaled,
            "all": recon_data,
            "rescaled_signal": np.sum(recon_scaled, axis=0, keepdims=True),
            "scaling_factor": scale_factors,
            "selected_freq": self.reduced["reduced_freqs"][:, :]
        }
