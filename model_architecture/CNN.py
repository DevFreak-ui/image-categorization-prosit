import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CINIC-10 image classification.
    Input: 32x32x3 images
    Output: 10 classes
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32x32x32 -> 32x32x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 32x32x64 -> 32x32x128
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        # So final feature map is 4x4x128 = 2048
        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 32x32x3 -> 16x16x32
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))  # 16x16x32 -> 8x8x64
        
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))  # 8x8x64 -> 4x4x128
        
        # Flatten for fully connected layers
        x = x.view(-1, 4 * 4 * 128)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class AdvancedCNN(nn.Module):
    """
    More advanced CNN with batch normalization and more layers.
    """
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # Convolutional blocks with batch normalization
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)  # 32x32x3 -> 16x16x64
        x = self.conv_block2(x)  # 16x16x64 -> 8x8x128
        x = self.conv_block3(x)  # 8x8x128 -> 4x4x256
        
        # Global average pooling
        x = self.global_avg_pool(x)  # 4x4x256 -> 1x1x256
        x = x.view(x.size(0), -1)    # Flatten to 256
        
        x = self.classifier(x)
        return x

def get_cnn_model(model_type='simple', num_classes=10):
    """
    Factory function to get CNN models.
    
    Args:
        model_type (str): 'simple' or 'advanced'
        num_classes (int): Number of output classes
    
    Returns:
        torch.nn.Module: CNN model
    """
    if model_type == 'simple':
        return SimpleCNN(num_classes)
    elif model_type == 'advanced':
        return AdvancedCNN(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'simple' or 'advanced'")

if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Simple CNN
    print("Testing Simple CNN...")
    simple_cnn = SimpleCNN().to(device)
    test_input = torch.randn(4, 3, 32, 32).to(device)
    output = simple_cnn(test_input)
    print(f"Simple CNN - Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Simple CNN - Parameters: {sum(p.numel() for p in simple_cnn.parameters()):,}")
    
    # Test Advanced CNN
    print("\nTesting Advanced CNN...")
    advanced_cnn = AdvancedCNN().to(device)
    output = advanced_cnn(test_input)
    print(f"Advanced CNN - Input shape: {test_input.shape}, Output shape: {output.shape}")
    print(f"Advanced CNN - Parameters: {sum(p.numel() for p in advanced_cnn.parameters()):,}")
