import torch
import torch.nn as nn

class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def get_cnn_model():
    return ModelCNN()