from scipy.ndimage import gaussian_filter1d

class SignalReconstructor:
    def __init__(self, recon_result, raw_signal, dn_weight, cutoff, fs):
        self.recon = recon_result
        self.raw = raw_signal
        self.dn_weight = dn_weight
        self.cutoff = cutoff
        self.fs = fs

    def reconstruct(self):
        recon = self.recon['mean_rescale'].real * self.dn_weight
        lp = lowpass_filter(self.raw, self.cutoff, self.fs)
        smooth_lp = gaussian_filter1d(lp, sigma=10)  # <- replace with your `filt_level` variable if needed
        return smooth_lp + recon
