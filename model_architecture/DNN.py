import torch
import torch.nn as nn

class VanillaDNN(nn.Module):
    """
    Vanilla Deep Neural Network (DNN) for CINIC-10 image classification.
    Input: 32x32x3 images (flattened to 3072)
    Output: 10 classes
    """
    def __init__(self, num_classes=10):
        super(VanillaDNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # Flatten to 3072
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepDNN(nn.Module):
    """
    Deeper DNN with more layers and batch normalization.
    """
    def __init__(self, num_classes=10):
        super(DeepDNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # Flatten to 3072
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

def get_dnn_model(model_type='vanilla', num_classes=10):
    """
    Factory function to get DNN models.
    
    Args:
        model_type (str): 'vanilla' or 'deep'
        num_classes (int): Number of output classes
    
    Returns:
        torch.nn.Module: DNN model
    """
    if model_type == 'vanilla':
        return VanillaDNN(num_classes)
    elif model_type == 'deep':
        return DeepDNN(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'deep'")

if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Vanilla DNN
    print("Testing Vanilla DNN...")
    vanilla_dnn = VanillaDNN().to(device)
    test_input = torch.randn(4, 3, 32, 32).to(device)
    output = vanilla_dnn(test_input)
    print(f"Vanilla DNN - Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Vanilla DNN - Parameters: {sum(p.numel() for p in vanilla_dnn.parameters()):,}")
    
    # Test Deep DNN
    print("\nTesting Deep DNN...")
    deep_dnn = DeepDNN().to(device)
    output = deep_dnn(test_input)
    print(f"Deep DNN - Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Deep DNN - Parameters: {sum(p.numel() for p in deep_dnn.parameters()):,}")
