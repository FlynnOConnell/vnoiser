import numpy as np
from .utils_st.signal_utils import trigger_detect
from .config import cwtReducerConfig
from scipy.signal import savgol_filter

class WaveletReducer:
    def __init__(self, cluster_result, coeff, freqs, scales, config: cwtReducerConfig = None):
        self.cluster_result = cluster_result
        self.coeff = coeff
        self.freqs = freqs
        self.scales = scales

        # use provide config or defaults
        self.config = config or cwtReducerConfig()

    def reduce_and_threshold(self):
        event_onsets = {}
        reduced_coeff, reduced_freq, reduced_scale = [], [], []
        selected_clu = []

        for i, freq_id in enumerate(np.unique(self.cluster_result[self.config.cluster_label])):
            freq_range = self.cluster_result[self.config.cluster_freq][freq_id]
            mask = np.isin(self.freqs, freq_range)
            if not np.any(mask):
                continue

            avg_coeff = np.mean(self.coeff[mask, :], axis=0)
            avg_scale = np.mean(self.scales[mask], axis=0)
            min_f, max_f = min(freq_range), max(freq_range)

            if max_f <= 15:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=41, polyorder=1)
            elif 15 < max_f <= 50:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=21, polyorder=1)
            elif 50 < max_f <= 75:
                avg_coeff = savgol_filter(avg_coeff.real, window_length=11, polyorder=1)

            # if max_f > self.config.cutoff_freq[1] or max_f <= self.config.cutoff_freq[0]:
            #     print(max_f)
            #     continue

            up_thresh, down_thresh = (
                (np.std(avg_coeff) * self.config.slow_upthres, np.std(avg_coeff)) if max_f < 30
                else (np.std(avg_coeff) * self.config.fast_upthres, 0)
            )

            crossings = trigger_detect(avg_coeff, up_thresh, down_thresh)
            if (crossings["Area"] == 0).all():
                continue

            iei = crossings["Area"][:,1] - crossings["Area"][:,0]
            event_onsets[f"clu{i}_{np.around(max_f,1)}Hz"] = {
                "crossings": crossings["Area"], "iei": iei, "num_event": len(iei)
            }

            reduced_coeff.append(avg_coeff)
            reduced_freq.append([min_f, max_f])
            reduced_scale.append(avg_scale)
            selected_clu.append(i)

        return {
            "reduced_coeff": np.array(reduced_coeff),
            "reduced_freqs": np.array(reduced_freq),
            "reduced_scale": np.array(reduced_scale),
            "event_onsets": event_onsets,
            "selected_clusters": np.array(selected_clu)
        }
