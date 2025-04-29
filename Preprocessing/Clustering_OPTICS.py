import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from tqdm import tqdm
import warnings

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 数据预处理
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']


train_data = pd.read_csv('adult.data', names=columns, na_values=' ?')
test_data = pd.read_csv('adult.test', names=columns, na_values=' ?', skiprows=1)

# 去除无效值

train_clean = train_data.dropna()
test_clean = test_data.dropna()

x_train = train_clean.drop('income', axis=1)
y_train = train_clean['income']

# 特征工程

cat_cols = x_train.select_dtypes(include=['object']).columns
num_cols = x_train.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

xtr_train = preprocessor.fit_transform(x_train)
xtr_train_dense = xtr_train.toarray() if hasattr(xtr_train, "toarray") else xtr_train
from sklearn.decomposition import PCA

# 在聚类前添加PCA降维
pca = PCA(n_components=0.95)  # 保留95%的方差
xtr_reduced = pca.fit_transform(xtr_train_dense)
print(f"降维后特征数：{xtr_reduced.shape[1]}")
# OPTICS聚类（带进度条）
clustering = OPTICS(
    min_samples=3,
    xi=0.03,
    min_cluster_size=15,
    metric='euclidean',  # 改为欧氏距离
    algorithm='auto',    # 让sklearn自动选择最佳算法
    n_jobs=-1
)
# 自定义进度条包装器
class OPTICSProgressWrapper:
    def __init__(self, optics_model):
        self.model = optics_model
        self.pbar = None
        
    def fit(self, X):
        with tqdm(total=100, desc="聚类进度") as self.pbar:
            self.model.fit(X)
            self.pbar.update(100 - self.pbar.n)
        return self

# 使用带进度条的聚类
clustering_wrapped = OPTICSProgressWrapper(clustering)
clustering_wrapped.fit(xtr_train_dense)
labels = clustering.labels_

# 评估结果

valid_mask = labels != -1
xtr_valid = xtr_train_dense[valid_mask]
labels_valid = labels[valid_mask]

if len(np.unique(labels_valid)) >= 2:
    silhouette = silhouette_score(xtr_valid, labels_valid)
    calinski = calinski_harabasz_score(xtr_valid, labels_valid)
    davies = davies_bouldin_score(xtr_valid, labels_valid)
    
    print(f"\n聚类评估结果:")
    print(f"轮廓系数(Silhouette Score): {silhouette:.4f} (越接近1越好)")
    print(f"Calinski-Harabasz指数: {calinski:.4f} (越大越好)")
    print(f"Davies-Bouldin指数: {davies:.4f} (越小越好)")
else:
    print("有效簇数量不足，无法计算评估指标")

# 显示聚类信息
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)
print(f"\n聚类完成，共发现 {n_clusters} 个簇，{n_noise} 个噪声点")

#聚类评估结果:
#轮廓系数(Silhouette Score): 0.3389 (越接近1越好)
#Calinski-Harabasz指数: 138.9311 (越大越好)
#Davies-Bouldin指数: 1.2211 (越小越好)

#聚类完成，共发现 339 个簇，23813 个噪声点
