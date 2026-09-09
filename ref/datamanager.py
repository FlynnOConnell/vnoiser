import os
import pickle
import h5py

class DataLoader:
    def __init__(self, data_path, roi_path, config: LoadConfig = None):
        self.data_path = data_path
        self.roi_path = roi_path

        self.config = config or LoadConfig()

    def load_raw(self):
        print(self.data_path)
        with open(self.data_path, 'rb') as f:
            ds_img = pickle.load(f, encoding='latin1')

        print()
        animalID = list(ds_img.keys())
        print(f"Animal ID is : {animalID}")
        exptID = list(ds_img[animalID[0]].keys())
        print(f"Experiment ID is : {exptID}")
        dataset = ds_img[animalID[0]][exptID[0]]
        scanID = list(dataset.keys())
        print(f"Included Scans : {scanID}")
        print(f"Scans include ROIs & Behavioral Data : {list(dataset[scanID[0]].keys())}")
        print('\nEach ROI includes ...')
        print(list(dataset[scanID[0]][0]['green'].keys()))

        return ds_img[animalID[0]][exptID[0]]

    def load_roi(self):
        # which scans was selected?
        with open(self.roi_path, 'rb') as f:
            scanIDs_ROIs = pickle.load(f, encoding='latin1')

        print(scanIDs_ROIs.keys())
        ROI_keys = list(scanIDs_ROIs.keys())
        #scanID_process = scanIDs_ROIs['scanID_SWR_pre_post']
        scanID_process = scanIDs_ROIs['scanID_spatial']
        print('Scan to process', scanID_process)
        domain_ROInumber = scanIDs_ROIs['domain_ROInumber']
        roi_list = scanIDs_ROIs['roi_list']
        print()
        for myroi in domain_ROInumber.keys():
            print(myroi, domain_ROInumber[myroi])

        return scanIDs_ROIs
