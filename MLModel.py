import torch
import torch.nn as nn
import torch.nn.functional as F

class MLModel:
    def __init__ (self, input_channels, num_conv_layers=2, out_channels=None,  kernel_sizes=None, num_fc_layers=2, fc_units=None, num_classes=50):
        
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        
        #super(MLModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        if out_channels is None:
            out_channels = [32] * num_conv_layers
        if kernel_sizes is None:
            kernel_sizes = [3] * num_conv_layers
        
        in_channels = input_channels
        for i in range (num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels[i], kernel_sizes[i]))
            in_channels = out_channels[i]
        
        sample_input = torch.randn(1, input_channels, 28, 28)
        self.flattened_size = self._calculate_flattened_size(sample_input)
        
        if fc_units is None:
            fc_units = [128] * num_fc_layers
        
        in_features = self.flattened_size
        for i in range (num_fc_layers):
            self.fc_layers.append(nn.Linear(in_features, fc_units[i]))
            in_features = fc_units[i]
            
        self.output_layer = nn.Linear(in_features, num_classes)
    
    def _calculate_flattened_size(self, sample_input):
        """Helper function to determine the flattened size after conv layers."""
        x = sample_input
        with torch.no_grad():
            for conv in self.conv_layers:
                x = conv(x)
                x = F.relu(x)  # Apply activation in forward pass
            flattened_size = x.view(x.size(0), -1).size(1)  # Flatten and get size
        return flattened_size

    def forward(self, x):
        # Pass through convolutional layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Final output layer (ReLu as this is a regretion based project)
        x = F.relu(self.output_layer(x))
        return x
