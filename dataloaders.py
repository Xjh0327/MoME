import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
import os

    
class Dataset(torch.utils.data.Dataset):
    """
    The framework of Dataset
    """

    def __init__(self, dataset_name, mode):
        folder = ''
        data = pd.read_csv(folder).to_numpy()[:, 1:]
        self.categorical_data = 
        self.numerical_data = 
        self.labels = 
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]
    
def load_data(batch_size, data_name):
    """
    The framework of Dataloader
    """
    num_classes = [1,1]
    kwargs = {'num_workers': 4, 'pin_memory': False}
    train_dataset = Dataset(data_name, mode='train')
    test_dataset = Dataset(data_name, mode='test')

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_classes, field_dims, numerical_num
