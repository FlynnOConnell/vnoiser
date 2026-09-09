import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from .config import DfofConfig
from .utils_st.signal_utils import calc_dfof_gauss

class GetDfof:
    def __init__(self, dataset, domain_ROInumber, scanID_process, config: DfofConfig = None):
        self.dataset = dataset
        self.domain_ROInumber = domain_ROInumber
        self.scanID_process = scanID_process

        self.config = config or DfofConfig()

    def run(self):
        img_scans = {}
        for myscan in self.scanID_process:
            #myroi_list = self.roi_list[myscan]
            mydataset = self.dataset[myscan]

            img_all = {}
            exclude_value = self.config.exclude_val
            for roi_name, roi_id in self.domain_ROInumber.items():
                if roi_name in exclude_value:
                    continue
                img_roi = np.array([])
                for myroi in roi_id:
                    if img_roi.size == 0:
                        img_roi = np.squeeze(mydataset[myroi]['green']['raw'])
                    else:
                        img_roi = np.concatenate((img_roi, np.squeeze(mydataset[myroi]['green']['raw'])), axis=1)
                img_all[roi_name] = img_roi.T
            img_scans[myscan] = img_all

        # if scans have different frame size
        # calculate dfof
        dfof_con_scans_mtx = {}
        for s, myscan in enumerate(self.scanID_process):
            myimg_all = img_scans[myscan]
            dfof_con = np.array([])
            for r, (myroi, myimg) in enumerate(myimg_all.items()):
                mysig_roi = np.mean(myimg, axis=0)
                mydfof = calc_dfof_gauss(mysig_roi,1500)
                if r == 0:
                    dfof_con = mydfof.reshape(-1,1).T
                    #dfof_con = mydfof
                else:
                    dfof_con = np.concatenate((dfof_con, mydfof.reshape(-1,1).T), axis=0)
                    #dfof_con = np.concatenate((dfof_con, mydfof), axis=0)
            dfof_con_scans_mtx[myscan] = dfof_con
            print('Scan',myscan, dfof_con_scans_mtx[myscan].shape)

        # z-score
        dfof_zscore_scans = {}
        for s, myscan in enumerate(self.scanID_process):
            print(f"applying ... {myscan}, {s+1} / {len(self.scanID_process)}")
            mydfof = dfof_con_scans_mtx[myscan]
            mydfof[:,0:1000] = np.mean(mydfof[:,1000:],axis=1, keepdims=True)
            baseline = gaussian_filter1d(mydfof, sigma=5000, axis=1)
            if self.config.negative == True:
                dfof_zscore_scans[myscan] = -1 * (mydfof - baseline) / np.std(mydfof, axis=1, keepdims=True)
            else:
                dfof_zscore_scans[myscan] = (mydfof - baseline) / np.std(mydfof, axis=1, keepdims=True)

        # save dfof
        #filename = 'stan95_2_dfof_data.h5'
        file2save = self.config.filepath + self.config.filename
        print(f"saving {self.config.filename} to {self.config.filepath}....")
        with h5py.File(file2save, 'w') as f:
            for myscan in self.scanID_process:
                grp = f.create_group(myscan)
                grp.create_dataset('dfof_raw', data=dfof_con_scans_mtx[myscan], compression='gzip', compression_opts=9)
                grp.create_dataset('dfof_zscore', data=dfof_zscore_scans[myscan], compression="gzip", compression_opts=9)

        print('dfof caluculations complete and saved')
        return {'raw': dfof_con_scans_mtx,
               'zscore': dfof_zscore_scans}
