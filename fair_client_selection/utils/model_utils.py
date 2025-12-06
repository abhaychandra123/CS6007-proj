"""Model definitions for federated learning experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for image classification (supports both grayscale and RGB)."""
    
    def __init__(self, num_classes=10, in_channels=3, input_size=32):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after 3 pooling layers
        # input_size -> input_size/2 -> input_size/4 -> input_size/8
        feature_size = (input_size // 8) * (input_size // 8) * 64
        
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_last_layer_params(self):
        """Get last layer parameters for Shapley value computation."""
        return self.fc2.weight.data.clone()


class MLPClassifier(nn.Module):
    """MLP for tabular data (Adult Census, etc.)."""
    
    def __init__(self, input_dim, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_last_layer_params(self):
        """Get last layer parameters for Shapley value computation."""
        return self.fc3.weight.data.clone()


def get_model(model_name, **kwargs):
    """Factory function to get model by name."""
    models = {
        'cnn': SimpleCNN,
        'mlp': MLPClassifier,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)


def get_parameters(model):
    """Extract model parameters as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Set model parameters from list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
