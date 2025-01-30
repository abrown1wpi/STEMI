import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dataprep(Dataset):
    def __init__(self, num_classes=8, window_size=0.1, sampling_rate=125):
        self.data=[]
        self.labels=[]
        samples_per_window = int(sampling_rate*window_size)
                
        