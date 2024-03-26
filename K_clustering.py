import torch
import torch.nn as nn
from resnet_he import ResNet1D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
original_model = ResNet1D(11,6,5,64)
original_model.load_state_dict(torch.load('resnet_model_cpu.pth'))

# 去除最后的线性层
change = nn.Sequential()
original_model.fc = change
final_CNN = original_model
print(final_CNN)

# 读取数据
original_data = pd.read_csv('wine_white.csv')
features = original_data.iloc[:, :11].values
labels = original_data.iloc[:, 11].values

# 转换为张量并移动到设备
device = "cuda" if torch.cuda.is_available() else "cpu"
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(-1).to(device)

# 提取特征
representation_features = final_CNN(features_tensor)

# 将张量转换为 NumPy 数组，并展平
representation_features = representation_features.detach().numpy()
representation_features = representation_features.reshape(representation_features.shape[0], -1)

# 使用 PCA 进行降维
pca = PCA(n_components=2)
pca_features = pca.fit_transform(representation_features)

# 使用 K-means 进行聚类
kmeans = KMeans(n_clusters=6)
kmeans.fit(representation_features)
cluster_labels = kmeans.labels_

# 可视化聚类效果
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y']

for cluster_id in range(6):
    plt.scatter(pca_features[cluster_labels == cluster_id, 0], pca_features[cluster_labels == cluster_id, 1],
                c=colors[cluster_id], label=f'Cluster {cluster_id}')

plt.title('Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
