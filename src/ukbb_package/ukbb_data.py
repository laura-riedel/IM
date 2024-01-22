# standard modules
import numpy as np
import pandas as pd
from scipy.stats import zscore
import math
import random

# PyTorch modules
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

# own utils module
from ukbb_package import utils

##################################################################################

class UKBBDataset(Dataset):
    """
    Dataset class that prepares ICA timeseries of rs-fMRI data from the UKBioBank.
    Input:
        data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        ica: whether to load ICA25 ('25') or ICA100 ('100'). Expects string. Default: 25.
        good_components: boolean flag to indicate whether to use only the good components or all components. Default: False.
        all_data: boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.
        index: whether an item is retrieved based on index (=True) or based on subject id (=False). Default: True.
    """
    def __init__(self, data_path, ica='25', good_components=False, all_data=True):
        # save data path + settings
        self.data_path = data_path
        self.ica = ica
        self.good_components = good_components
        self.all_data = all_data
        self.additional_data = '../../data/'

        # catch ica error
        if ica != '25' and ica != '100':
            # ERROR MESSAGE ODER SO
            pass

        # get indices of good components
        if good_components:
            if ica == '25':
                self.good_components_idx = np.loadtxt(self.additional_data+'rfMRI_GoodComponents_d25_v1.txt', dtype=int)
            else:
                self.good_components_idx = np.loadtxt(self.additional_data+'rfMRI_GoodComponents_d100_v1.txt', dtype=int)
            # component numbering starts at 1, not at 0
            self.good_components_idx = self.good_components_idx-1
            
        # META INFORMATION
        # get target information (age)
        age_df = pd.read_csv(self.data_path+'table/targets/age.tsv', sep='\t', names=['age'])
        # get subject IDs, rename column for later merge
        ids_df = pd.read_csv(self.data_path+'table/ukb_imaging_filtered_eids.txt')
        ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
        # get location of ica files
        location_df = pd.read_csv(self.additional_data+'ica_locations.csv')
        # combine information
        # first ids and age
        meta_df = pd.concat([ids_df, age_df], axis=1)
        # merge with location info based on eid
        self.location_column = 'rfmri_ica'+self.ica+'_ts'
        meta_df = meta_df.merge(location_df[['eid',self.location_column]], on='eid', how='left')
        # limit df to available ICA25 data points
        meta_df = meta_df[meta_df[self.location_column].isna()==False]
        # reset index
        meta_df = meta_df.reset_index(drop=True, inplace=False)
        
        # reduce amount of data for debugging etc.
        if not self.all_data:
            meta_df = meta_df[0:100]
        
        # save meta_df information as labels
        self.labels = meta_df
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, sub_id):       
        # get label (age)
        label = self.labels.loc[self.labels['eid'] == sub_id, 'age'].values[0]
        # get filname/path to timeseries
        ts_path = self.labels.loc[self.labels['eid'] == sub_id,self.location_column].values[0]
        
        # load + standardise timeseries
        ## if good_components = True, only load good components
        if self.good_components:
            timeseries = np.loadtxt(ts_path, 
                                    usecols=tuple(self.good_components_idx),
                                    max_rows=490) # crop longer input to 490 rows
        else:
            timeseries = np.loadtxt(ts_path, 
                                    max_rows=490) # crop longer input to 490 rows
        ## change axes ([components, time points])
        timeseries = np.swapaxes(timeseries, 1, 0)
        ## 
        ## standardise each component's timeseries
        timeseries = zscore(timeseries, axis=1)
        
        # turn data into tensors
        timeseries = torch.from_numpy(timeseries)
        timeseries = timeseries.float()
        label = torch.tensor(label)
        label = label.float()
        return timeseries, label, sub_id        
    
class UKBBDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning style DataModule class that prepares 
    & loads ICA timeseries of rs-fMRI data from the UKBioBank.
    """
    def __init__(self, data_path, ica='25', good_components=False, all_data=True, batch_size=128, seed=43): 
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.ica = ica
        self.good_components = good_components
        self.all_data = all_data
        self.batch_size = batch_size
        self.seed = seed
        # increase reproducibility by setting a generator 
        self.g = torch.Generator() # device='cuda'
        self.g.manual_seed(seed)
        utils.make_reproducible(seed)
        
    def get_split_index(self, split_ratio, dataset_size):
        return int(np.floor(split_ratio * dataset_size))
   
    def setup(self, stage=None):
        utils.make_reproducible(self.seed)
        # runs on all GPUs
        # load data
        self.data = UKBBDataset(self.data_path, self.ica, self.good_components, self.all_data)
        
        # split data
        dataset_size = self.data.labels['eid'].shape[0]
        indices = list(self.data.labels['eid'])            
        
        # shuffle
        np.random.shuffle(indices)
        indices = list(indices)
        # get list of split indices, feed to samplers
        split_index = self.get_split_index(0.8, dataset_size)
        self.train_idx, remain_idx = indices[:split_index], indices[split_index:]
        remain_split_index = self.get_split_index(0.5, len(remain_idx))
        self.val_idx, self.test_idx = remain_idx[:remain_split_index], remain_idx[remain_split_index:]
        
        assert len(self.val_idx)+len(self.test_idx) == len(self.val_idx+self.test_idx), 'Idx addition error'
        
        self.train_sampler = SubsetRandomSampler(self.train_idx, generator=self.g)
        self.val_sampler = SubsetRandomSampler(self.val_idx, generator=self.g)
        self.test_sampler = SubsetRandomSampler(self.test_idx, generator=self.g)
        self.predict_sampler = SubsetRandomSampler(self.val_idx+self.test_idx, generator=self.g)
    
    def general_dataloader(self, data_part, data_sampler):
        data_loader = DataLoader(
                        data_part,
                        batch_size = self.batch_size,
                        sampler = data_sampler,
                        pin_memory = True, ##
                        num_workers = 0, ##
                        # drop_last=False,
                        worker_init_fn = utils.seed_worker,
                        generator = self.g,
                        )
        return data_loader
        
    def train_dataloader(self):
        return self.general_dataloader(self.data, self.train_sampler)
    
    def val_dataloader(self):
        return self.general_dataloader(self.data, self.val_sampler)
    
    def test_dataloader(self):
        return self.general_dataloader(self.data, self.test_sampler)
    
    def predict_dataloader(self):
        return self.general_dataloader(self.data, self.predict_sampler)
