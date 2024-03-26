
# training for ResNet（Residual Network）
# data : 1D-data : wine-white.csv
# data features:column 1-11
# data labels:column 12
# the number of samples is : 4898,which is able to train a residual network.
# data procession : transform to be 1-D tensor:(n_samples,in_channels,length)

# data procession
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

original_data = pd.read_csv('wine_white.csv')
features = original_data.iloc[:, :11].values
labels = original_data.iloc[:, 11].values


features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

features_tensor = features_tensor.unsqueeze(-1).to(device)

from resnet_he import ResNet1D
model = ResNet1D(11,6,5,64)
model = model.to(device)

# 定义批次大小和训练迭代次数
batch_size = 64
epochs = 150
optimizer = optim.Adam(params=model.parameters(), lr=0.001, betas= (0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.CrossEntropyLoss()
# 训练模型
print('-------------------Start Training!-------------------')

for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(features_tensor), batch_size):
        inputs = features_tensor[i:i + batch_size]
        targets = labels_tensor[i:i + batch_size]

        # 梯度清零
        optimizer.zero_grad()

        # 正向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss}")

torch.save(model.state_dict(), 'resnet_model_cpu.pth')

# testing data
test_data = pd.read_csv('wine_red.csv')
test_data_features = test_data.iloc[:,:11].values
test_data_labels = test_data.iloc[:,11].values
test_data_features_tensor = torch.tensor(test_data_features,dtype=torch.float32)
test_data_labels_tensor = torch.tensor(test_data_labels,dtype=torch.long).to(device)

test_data_features_tensor = test_data_features_tensor.unsqueeze(-1).to(device)
model.eval()

with torch.no_grad():
    outputs = model(test_data_features_tensor)
    _, predicted = torch.max(outputs, 1)

correct = (predicted == test_data_labels_tensor).sum().item()
total = test_data_labels_tensor.size(0)
accuracy = correct / total
print(f'Accuracy: {accuracy}')


