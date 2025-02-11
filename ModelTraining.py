import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dataprep(Dataset):
    def __init__(self, num_classes=15, window_size=0.1, sampling_rate=250):
        self.tempData=[]
        self.tempLabels=[]
        samples_per_window = int(sampling_rate*window_size)
        
        for i in range(0, num_classes):
            curData = np.load(f"emg_datasets{i}.npy")
            
            num_windows = curData.shape[1] // samples_per_window
            
            for j in range(num_windows):
                firstVal = j * samples_per_window
                endVal = firstVal + samples_per_window
                
                dataseg = curData[:, firstVal:endVal]
                
                self.tempData.append(dataseg)
                # append data label
                match i:
                    case 0:
                        self.tempLabels.append([0,0,1,0,0])
                    case 1:
                        self.tempLabels.append([0,1,1,1,0])
                    case 2:
                        self.tempLabels.append([0,0,0,0,1])
                    case 3:
                        self.tempLabels.append([0,0,0,0,0.5])
                    case 4:
                        self.tempLabels.append([1,0,1,1,1])
                    case 5:
                        self.tempLabels.append([0,1,1,1,1])
                    case 6:
                        self.tempLabels.append([0,0,0,0,0])
                    case 7:
                        self.tempLabels.append([0,1,1,0,0])
                    case 8:
                        self.tempLabels.append([1,0.5,0.5,0.5,0.5])
                    case 9:
                        self.tempLabels.append([1,1,0,0,1])
                    case 10:
                        self.tempLabels.append([0,0,1,1,1])
                    case 11:
                        self.tempLabels.append([0.5,0,0,0,0])
                    case 12:
                        self.tempLabels.append([1,1,0,0,0])
                    case 13:
                        self.tempLabels.append([0,0,0.5,0.5,0.5])
                    case 14:
                        self.tempLabels.append([0,0,1,1,1])
                    case _:
                        print("Failue.")
        
        self.data = np.array(self.tempData, dtype=np.float32)
        self.labels = np.array(self.tempLabels, dtype=np.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])




import torch.nn as nn
import torch.nn.functional as F

class MLModel(nn.Module):
    def __init__ (self, input_channels, num_conv_layers=2, out_channels=None,  kernel_sizes=None, num_fc_layers=2, fc_units=None):
        super(MLModel, self).__init__()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        
        if out_channels is None:
            out_channels = [32] * num_conv_layers
        if kernel_sizes is None:
            kernel_sizes = [3] * num_conv_layers

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        in_channels = input_channels
        for i in range (num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels[i], kernel_sizes[i]))
            in_channels = out_channels[i]
        
        sample_input = torch.randn(1, input_channels, 8, 12)
        self.flattened_size = self._calculate_flattened_size(sample_input)
        
        if fc_units is None:
            fc_units = [128] * num_fc_layers
        
        in_features = self.flattened_size
        for i in range (num_fc_layers):
            self.fc_layers.append(nn.Linear(in_features, fc_units[i]))
            in_features = fc_units[i]
            
        self.output_layer = nn.Linear(in_features, 5)
    
    def _calculate_flattened_size(self, sample_input):
        """Helper function to determine the flattened size after conv layers."""
        x = sample_input
        with torch.no_grad():
            for conv in self.conv_layers:
                x = F.relu(conv(x))  # Apply activation in forward pass
            flattened_size = x.view(x.size(0), -1).size(1)  # Flatten and get size
        return flattened_size

    def forward(self, x):
        x = x.unsqueeze(-1)  # Change to [batch_size, 8, 12, 1]
        x = x.permute(0, 3, 1, 2)  # Rearrange to [batch_size, 1, 8, 12]

        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.reshape(x.size(0), -1)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.output_layer(x)



import torch.optim as optim
def training_loop():
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Still using MSE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)  # Ensure correct dtype

            optimizer.zero_grad()
            outputs = model(inputs)  # Outputs shape: (batch_size, 5)
            loss = criterion(outputs, targets)  # Compute MSE loss over all 5 values
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
        
dataset = Dataprep()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model=MLModel(input_channels=1)
training_loop()
