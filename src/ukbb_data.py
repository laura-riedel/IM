# standard modules
import numpy as np
import pandas as pd
from scipy.stats import zscore
import math

# PyTorch modules
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

# own utils module
import utils

# (f)MRI modules
# import bids

##################################################################################

class UKBBDataset(Dataset):
    """
    Dataset class that prepares ICA25 timeseries of rs-fMRI data from the UKBioBank.
    """
    def __init__(self, data_path, all_data=True):
        # save data path
        self.data_path = data_path
        self.all_data = all_data
        
        # META INFORMATION
        # get target information (age)
        age_df = pd.read_csv(self.data_path+'table/targets/age.tsv', sep='\t', names=['age'])
        # get subject IDs, rename column for later merge
        ids_df = pd.read_csv(self.data_path+'table/ukb_imaging_filtered_eids.txt')
        ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
        # get location of ica files
        location_df = pd.read_csv(self.data_path+'table/files_table.csv')
        # combine information
        # first ids and age
        meta_df = pd.concat([ids_df, age_df], axis=1)
        # merge with location info based on eid
        meta_df = meta_df.merge(location_df[['eid','rfmri_ica25_ts']], on='eid', how='left')
        # limit df to available ICA25 data points
        meta_df = meta_df[meta_df['rfmri_ica25_ts'].isna()==False]
        # reset index
        meta_df = meta_df.reset_index(drop=True, inplace=False)
        
        # reduce amount of data for debugging etc.
        if not self.all_data:
            meta_df = meta_df[0:100]
        
        # save meta_df information as labels
        self.labels = meta_df
        
        # pybids version
        # BIDS LAYOUT
        # self.layout = bids.BIDSLayout(self.data_path+'bids/', validate=False)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # get subject ID
        sub_id = self.labels.loc[idx,'eid']
        # get label (age)
        label = self.labels.loc[idx,'age']
        #label = int(label)
        
        # get filename/path to timeseries
        ts_path = self.labels.loc[idx,'rfmri_ica25_ts']
        # pybids version
        #ts_path = self.layout.get(datatype='func',
        #                            subject=sub_id, 
        #                            task='rest',
        #                            session='2',
        #                            suffix='25', # filters for ts-ica-25
        #                            extension='txt',
        #                            return_type='filename')
        
        # load + standardise timeseries
        ## crop longer input to 490 rows
        timeseries = np.loadtxt(ts_path, max_rows=490)
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
        return timeseries, label        
    
class UKBBDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning style DataModule class that prepares 
    & loads ICA25 timeseries of rs-fMRI data from the UKBioBank.
    """
    def __init__(self, data_path, all_data=True, batch_size=128, seed=43): 
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.all_data = all_data
        self.batch_size = batch_size
        # increase reproducibility by setting a generator 
        self.g = torch.Generator() # device='cuda'
        self.g.manual_seed(seed)
        utils.make_reproducible(seed)
        
    def get_split_index(self, split_ratio, dataset_size):
        return int(np.floor(split_ratio * dataset_size))
   
    def setup(self, stage=None):
        # runs on all GPUs
        # load data
        self.data = UKBBDataset(self.data_path, self.all_data)
        
        # split data      
        # create indices
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        # shuffle
        np.random.shuffle(indices)
        # get list of split indices, feed to samplers
        split_index = self.get_split_index(0.8, dataset_size)
        self.train_idx, remain_idx = indices[:split_index], indices[split_index:]
        remain_split_index = self.get_split_index(0.5, len(remain_idx))
        self.val_idx, self.test_idx = remain_idx[:remain_split_index], remain_idx[remain_split_index:]
        
        self.train_sampler = SubsetRandomSampler(self.train_idx, generator=self.g)
        self.val_sampler = SubsetRandomSampler(self.val_idx, generator=self.g)
        self.test_sampler = SubsetRandomSampler(self.test_idx, generator=self.g)
    
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
