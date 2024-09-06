import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random


class Pointnet_Dataset(Dataset):
    def __init__(self, data_folder_path: str, is_train: bool = True, 
                 num_points: int = 361, cache: bool = False,
                 split: float = 0.8):
        self.data_folder_path = data_folder_path
        self.data_files_path_az = [os.path.join(data_folder_path, 'az', f) for f in os.listdir(os.path.join(data_folder_path, 'az')) if f.endswith('.csv')]
        self.data_files_path.sort()
        self.data_files_path_vt = [f.replace("az", "vt") for f in self.data_files_path_az]
        self.is_train = is_train
        self.split = split * len(self.data_files_path)
        self.data_files_path = self.data_files_path[:int(self.split)] if is_train else self.data_files_path[int(self.split):]
        
        self.num_points = num_points

        self.data = []
        self.cache = cache
        if cache:
            self.load_data()
    
    def load_data(self):
        for f in self.data_files_path:
            data = pd.read_csv(f, header=None).to_numpy()
            self.data.append(data)
    
    def __len__(self) -> int:
        return len(self.data_files_path)
    
    def __getitem__(self, index):
        if self.cache:
            data = self.data[index]
        else:
            data_az = pd.read_csv(self.data_files_path_az[index], header=None).to_numpy()
            data_vt = pd.read_csv(self.data_files_path_vt[index], header=None).to_numpy()
            data = np.concatenate((data_az, data_vt), axis=0)
        
        # Randomly selecting one point from every 10 points in data
        indices_1 = [random.randint(i*10, min((i+1)*10 - 1, data.shape[0]-1)) for i in range(data.shape[0] // 10)]
        indices_2 = [random.randint(i*10, min((i+1)*10 - 1, data.shape[0]-1)) for i in range(data.shape[0] // 10)]
        data_sampled_1 = data[indices_1, :]
        data_sampled_2 = data[indices_2, :]
        
        return data_sampled_1, data_sampled_2
    
    @staticmethod
    def collate_fn(batch):
        data_sampled_1, data_sampled_2 = list(zip(*batch))
        data_sampled_1 = torch.tensor(np.array(data_sampled_1)).float()
        data_sampled_2 = torch.tensor(np.array(data_sampled_2)).float()
        return data_sampled_1, data_sampled_2
