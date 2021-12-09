import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_dataset, val_dataset, test_dataset,  batch_size, shuffle, num_workers,
                 collate_fn=default_collate, pin_memory=True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.batch_idx = 0
        self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }
        super().__init__(**self.init_kwargs)

        self.n_samples = len(self.train_dataset)

        self.valid_data_loader_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True
        }

        self.valid_data_loader = DataLoader(**self.valid_data_loader_init_kwargs)

        self.valid_data_loader.n_samples = len(self.val_dataset)

        if self.test_dataset:

            self.test_data_loader_init_kwargs = {
                'dataset': self.test_dataset,
                'batch_size': batch_size,
                'shuffle': False,
                'collate_fn': collate_fn,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'drop_last': True
            }

            self.test_data_loader = DataLoader(**self.test_data_loader_init_kwargs)

            self.test_data_loader.n_samples = len(self.test_dataset)