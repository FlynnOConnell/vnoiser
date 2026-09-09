# signal_utils.py

import numpy as np
from scipy.ndimage import gaussian_filter1d

def trigger_detect(signal, up_thresh, down_thresh):
    signal = np.array(signal).flatten()
    len_signal = len(signal)
    pre_val = np.concatenate(([(up_thresh + down_thresh) / 2], signal[:-1]))

    up_crossings = np.where((pre_val < up_thresh) & (signal >= up_thresh))[0]
    down_crossings = np.where((pre_val > down_thresh) & (signal <= down_thresh))[0]

    up_down_unsort = np.concatenate((up_crossings, down_crossings))
    type_unsort = np.concatenate((np.ones(len(up_crossings)), -np.ones(len(down_crossings))))

    sorted_indices = np.argsort(up_down_unsort)
    up_down_sort = up_down_unsort[sorted_indices]
    type_sort = type_unsort[sorted_indices]

    prev_type = np.concatenate(([np.nan], type_sort[:-1]))
    crossings_up = up_down_sort[(prev_type == -1) & (type_sort == 1)]
    crossings_down = up_down_sort[(prev_type == 1) & (type_sort == -1)]

    crossings = {'Up': crossings_up, 'Dwn': crossings_down}

    if len(crossings_up) == 1 and len(crossings_down) == 1:
        if crossings_up < crossings_down:
            crossings['Area'] = np.array([[crossings_up[0], crossings_down[0]]])
        else:
            crossings['Area'] = np.array([[1, crossings_up[0]], [crossings_down[0], len_signal - 1]]).T
    elif len(crossings_up) + len(crossings_down) == 0:
        crossings['Area'] = np.array([]).reshape(0, 0)
    elif len(crossings_up) == 0:
        crossings['Area'] = np.array([1, crossings_down[0]])
    elif len(crossings_down) == 0:
        crossings['Area'] = np.array([crossings_up[0], len_signal - 1])
    else:
        if crossings_up[-1] > crossings_down[-1] and crossings_up[0] > crossings_down[0]:
            crossings['Dwn'] = np.append(crossings_down, len_signal - 1)
            crossings['Up'] = np.append([1], crossings_up)
        elif crossings_up[-1] > crossings_down[-1] and crossings_up[0] < crossings_down[0]:
            crossings['Dwn'] = np.append(crossings_down, len_signal - 1)
        elif crossings_up[-1] < crossings_down[-1] and crossings_up[0] > crossings_down[0]:
            crossings['Up'] = np.append([1], crossings_up)
        crossings['Area'] = np.vstack((crossings['Up'], crossings['Dwn'])).T

    return crossings

def PSTH_Basic(mysignals, timewin, onsets):
    mypsth = []
    for tt in onsets:
        start_idx = max(tt - timewin, 0)
        end_idx = min(tt + timewin + 1, len(mysignals))  # +1 to include the end index
        mywave = mysignals[start_idx:end_idx]

        # Pad with NaNs if the segment is shorter than expected (happens at the edges of the signal)
        if len(mywave) < 2 * timewin + 1:
            padding = 2 * timewin + 1 - len(mywave)
            mywave = np.pad(mywave, (0, padding), 'constant', constant_values=np.mean(mywave))

        mypsth.append(mywave)

    # Convert the list of arrays into a 2D NumPy array
    mypsth = np.array(mypsth)

    return mypsth

def centerize_psth(mypsth,stim_time,extra_bin,trim_bin):
    n_event, n_bin, n_roi = np.shape(mypsth)
    # pick-up soma psth
    mypsth_sm = gaussian_filter1d(mypsth[:,:,0], axis=1, sigma=3)

    psth_shifted = np.array([])
    for i in range(n_event):
        # detect peak time within a given window from stim time
        mypsth_sm_trim = mypsth_sm[i,stim_time-30:(stim_time+extra_bin)]
        mypeak = np.max(mypsth_sm_trim)
        mypeak_bin = np.where(mypsth_sm_trim==mypeak)[0]
        detected_peak = mypeak_bin + stim_time-30

        # shift psth based on the peak time
        st_shift = detected_peak-trim_bin
        ed_shift = detected_peak+trim_bin
        mypsth_re = mypsth[i,st_shift[0]:ed_shift[0],:]

        if len(psth_shifted) == 0:
            psth_shifted = mypsth_re[np.newaxis,:,:]
        else:
            psth_shifted = np.concatenate((psth_shifted, mypsth_re[np.newaxis,:,:]), axis=0)

    return psth_shifted

def calc_dfof_gauss(mydata, mysigma):
    baseline = gaussian_filter1d(mydata, sigma=mysigma, axis=0)
    dfof = (mydata - baseline) / baseline

    return dfof