import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
from tqdm import tqdm  # 进度条库
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


#第一步 data propossing

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']


train_data = pd.read_csv('adult.data', names=columns, na_values=' ?') ## 将？替换为NaN
test_data = pd.read_csv('adult.test', names=columns, na_values=' ?', skiprows=1)

#去除无效值
train_clean = train_data.dropna()
test_clean = test_data.dropna()

x_train, y_train = train_clean.drop('income', axis=1), train_clean['income']
x_test, y_test = test_clean.drop('income', axis=1), test_clean['income']


cat_cols = train_clean.drop('income', axis=1).select_dtypes(include=['object']).columns
num_cols = train_clean.select_dtypes(include=['number']).columns
#标准化+独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

xtr_train = preprocessor.fit_transform(x_train)
xtr_test = preprocessor.transform(x_test)
xtr_train_dense = xtr_train.toarray()
# 数据预处理部分保持不变...

# 初始化HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    core_dist_n_jobs=-1  # 使用所有CPU核心
)

# 添加进度条
print("Running HDBSCAN clustering...")
with tqdm(total=100) as pbar:
    # HDBSCAN没有直接提供进度回调，所以用模拟进度
    cluster_labels = clusterer.fit_predict(xtr_train_dense)
    pbar.update(100)  # 完成后更新到100%

def evaluate_clustering(labels, data):
    # 只评估非噪声点
    mask = labels != -1
    if len(set(labels[mask])) < 2:  # 至少需要2个簇才能计算
        return None, None, None
    
    data_valid = data[mask]
    labels_valid = labels[mask]
    
    silhouette = silhouette_score(data_valid, labels_valid)
    calinski = calinski_harabasz_score(data_valid, labels_valid)
    davies = davies_bouldin_score(data_valid, labels_valid)
    return silhouette, calinski, davies

# 评估
print("\nEvaluating results...")
silhouette, calinski, davies = evaluate_clustering(cluster_labels, xtr_train_dense)

print("\nHDBSCAN 评估结果:")
print(f"Silhouette Score: {silhouette}")
print(f"Calinski-Harabasz Index: {calinski}") 
print(f"Davies-Bouldin Index: {davies}")

#Silhouette Score: 0.21337778738417254
#Silhouette Score: 0.21337778738417254
#Calinski-Harabasz Index: 198.5375208597199
#Davies-Bouldin Index: 1.4582870120175473
