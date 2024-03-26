import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks=5, base_filters=64):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = self._make_blocks(num_blocks, base_filters)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def _make_blocks(self, num_blocks, base_filters):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(BasicBlock(base_filters, base_filters))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage:
# Define input size and number of classes
in_channels = 1  # Assuming input has one channel
num_classes = 10  # Number of output classes

# Create an instance of ResNet1D
model = ResNet1D(in_channels, num_classes)

# Forward pass
input_data = torch.randn(1, in_channels, 100)  # Assuming input sequence length is 100
output = model(input_data)

print("Output shape:", output.shape)  # Output shape should be (batch_size, num_classes)
