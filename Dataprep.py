import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data = np.load(f"emg_datasets1.npy")
print(data.shape[1])

class Dataprep(Dataset):
    def __init__(self, num_classes=8, window_size=0.1, sampling_rate=125):
        self.data=[]
        self.labels=[]
        samples_per_window = int(sampling_rate*window_size)
        
        for i in range(0, num_classes):
            curData = np.load(f"emg_dataset{i}.npy")
            
            num_windows = curData.shape[1] // samples_per_window
            
            for j in range(num_windows):
                firstVal = j * samples_per_window
                endVal = firstVal + samples_per_window
                
                dataseg = curData[:, firstVal:endVal]
                
                self.data.append(dataseg)
                # append data label
            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.float32)
        