import os
import pandas as pd
from torch.utils.data import Dataset
from torch import tensor

labs = dict({'even': 0, 'odd': 1})

class NumberDataset(Dataset):
    def __init__(self, dataset_file):
        self.numbers = dataset_file['number']
        self.labels = dataset_file['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.numbers.iloc[idx], self.labels.iloc[idx]