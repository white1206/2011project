import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 数据预处理部分保持不变...

# 改进的评估函数
def evaluate_clustering(labels, data):
    # 排除噪声点（标签为-1）
    valid_mask = labels != -1
    if sum(valid_mask) == 0:
        print("所有点都被识别为噪声！")
        return None, None, None
    
    if len(np.unique(labels[valid_mask])) < 2:
        print("有效簇数量不足（需要至少2个簇）")
        return None, None, None
    
    data_valid = data[valid_mask]
    labels_valid = labels[valid_mask]
    
    silhouette = silhouette_score(data_valid, labels_valid)
    calinski = calinski_harabasz_score(data_valid, labels_valid)
    davies = davies_bouldin_score(data_valid, labels_valid)
    
    return silhouette, calinski, davies

# 降维处理（关键改进）
pca = PCA(n_components=0.95)  # 保留95%方差
xtr_reduced = pca.fit_transform(xtr_train.toarray())

# DBSCAN参数优化
dbscan = DBSCAN(
    eps=3.0,            # 增大eps以适应标准化后数据
    min_samples=5,
    metric='euclidean',
    n_jobs=-1
)

dbscan_labels = dbscan.fit_predict(xtr_reduced)

# 评估并打印详细信息
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = sum(dbscan_labels == -1)
print(f"\n发现 {n_clusters} 个簇，{n_noise} 个噪声点（占总样本 {(n_noise/len(dbscan_labels)):.1%}）")

scores = evaluate_clustering(dbscan_labels, xtr_reduced)
if scores[0] is not None:
    print("\nDBSCAN 评估结果:")
    print(f"Silhouette Score: {scores[0]:.4f} (越接近1越好)")
    print(f"Calinski-Harabasz Index: {scores[1]:.1f} (越大越好)") 
    print(f"Davies-Bouldin Index: {scores[2]:.4f} (越小越好)")

# 可视化分析（可选）
import matplotlib.pyplot as plt
plt.scatter(xtr_reduced[:, 0], xtr_reduced[:, 1], c=dbscan_labels, cmap='viridis', s=10)
plt.colorbar(label='Cluster')
plt.title("DBSCAN Clustering (First Two PCA Components)")
plt.show()
