import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List

class MNISTDataset(Dataset):
    def __init__(
        self,
        path_to_data:str,
        test:bool = False,
        dataset:Dataset=None,
        **kwargs
    ):
        super().__init__()
        with open(path_to_data, 'rb') as f:
            self.dataset = pickle.load(f)
        
        self.num_pairs = self.dataset[0]
        self.labels = torch.nn.functional.one_hot(
            self.dataset[1],
            19
        )
        
    def __len__(self,):
        return len(self.num_pairs)
    
    def __getitem__(self, idx):
        return self.num_pairs[idx].float(), self.labels[idx].float()

class MyDataset(Dataset):
    
    def __init__(
        self,
        path_to_data:str,
        num_labels:int ,
        **kwargs
    ):
        super().__init__()
        with open(path_to_data, 'rb') as f:
            self.dataset = pickle.load(f)
        
        self.input = self.dataset[0]
        self.labels = self.dataset[1]
    
    def __len__(self,):
        return len(self.num_pairs)
    
    def __getitem__(self, idx):
        return self.input[idx].float(), self.labels[idx].float()




"""
train_dataset = MNISTDataset(train_data, test=False)
val_dataset = MNISTDataset(valid_data, test=False)
test_dataset = MNISTDataset(test_data, test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
"""    