
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN,OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
    

# 数据预处理函数
def preprocess_data():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']
    
    train_data = pd.read_csv('adult.data', names=columns, na_values=' ?')
    test_data = pd.read_csv('adult.test', names=columns, na_values=' ?', skiprows=1)
    #无效数据设置为NAN
    train_clean = train_data.dropna()
    test_clean = test_data.dropna()
    
    x_train = train_clean.drop('income', axis=1)
    y_train = train_clean['income']
    
    cat_cols = x_train.select_dtypes(include=['object']).columns
    num_cols = x_train.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols)
        ])
    
    xtr_train = preprocessor.fit_transform(x_train)
    return xtr_train


# 聚类评估函数(修改后对结果影响不大，只是严谨了点)
def evaluate_clustering(labels, data):
    # 排除噪声点（标签为-1）
    valid_mask = labels != -1
    if sum(valid_mask) == 0:
        print("All points are nosiy")
        return None, None, None
    
    if len(np.unique(labels[valid_mask])) < 2:
        print("Not enough valid clusters")
        return None, None, None
    
    data_valid = data[valid_mask]
    labels_valid = labels[valid_mask]
    
    silhouette = silhouette_score(data_valid, labels_valid)
    calinski = calinski_harabasz_score(data_valid, labels_valid)
    davies = davies_bouldin_score(data_valid, labels_valid)
    
    return silhouette, calinski, davies


# 聚类可视化函数（单独显示每个算法）
def plot_cluster_results(data_2d, labels, algorithm_name):
    plt.figure(figsize=(10, 8))
    
    # 为DBSCAN特别处理噪声点显示
    if -1 in labels:
        palette = sns.color_palette("viridis", n_colors=len(np.unique(labels)))
        # 将噪声点(-1)显示为灰色
        palette[0] = (0.7, 0.7, 0.7)
        hue_order = sorted(np.unique(labels))
    else:
        palette = 'viridis'
        hue_order = None
    
    scatter = sns.scatterplot(
        x=data_2d[:, 0], y=data_2d[:, 1], 
        hue=labels, palette=palette, hue_order=hue_order,
        alpha=0.8, s=50, edgecolor='w', linewidth=0.5
    )
    
    plt.title(f'{algorithm_name} Clustering (t-SNE projection)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # 调整图例位置和样式
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        title='Cluster',
        frameon=False
    
    )
    plt.show()
    

# 主函数
def main():
   
    # 数据预处理
    xtr_train = preprocess_data()
    xtr_train_dense = xtr_train.toarray()
    
    # t-SNE降维用于可视化（所有方法使用相同的降维结果）
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    xtr_tsne = tsne.fit_transform(xtr_train_dense)
    
    # 存储结果
    results = {}
    
    #1.k-means
    pca = PCA(n_components=20, random_state=42)
    xtr_pca = pca.fit_transform(xtr_train_dense)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(xtr_pca)  # 或用原始数据 xtr_train_dense
    results['K-means'] = {
        'labels': kmeans_labels,
        'scores': evaluate_clustering(kmeans_labels, xtr_train_dense)
    }
    plot_cluster_results(xtr_tsne, kmeans_labels, "K-means")
    
    
    # 2. Hierarchy
    pca = PCA(n_components=20)
    xtr_reduced = pca.fit_transform(xtr_train_dense)
    hierarchical = AgglomerativeClustering(
        n_clusters=2,
        metric='euclidean',  
        linkage='ward'  
    )
    hierarchical_labels = hierarchical.fit_predict(xtr_reduced)
    results['Hierarchical'] = {
        'labels': hierarchical_labels,
        'scores': evaluate_clustering(hierarchical_labels, xtr_train_dense)
    }
    plot_cluster_results(xtr_tsne, hierarchical_labels, "Hierarchical")
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=2.3, min_samples=5)  # 调整后的参数
    dbscan_labels = dbscan.fit_predict(xtr_train_dense)
    results['DBSCAN'] = {
        'labels': dbscan_labels,
        'scores': evaluate_clustering(dbscan_labels, xtr_train_dense)
    }
    plot_cluster_results(xtr_tsne, dbscan_labels, "DBSCAN")
    
    
    # 4. GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(xtr_train_dense)
    results['GMM'] = {
        'labels': gmm_labels,
        'scores': evaluate_clustering(gmm_labels, xtr_train_dense)
    }
    plot_cluster_results(xtr_tsne, gmm_labels, "Gaussian Mixture Model")

    #5. HDBSCAN
    hdbscan = HDBSCAN(
    min_cluster_size=8,
    min_samples=5,
    metric='euclidean',
    core_dist_n_jobs=-1 )
    hdbscan_labels=hdbscan.fit_predict(xtr_train_dense)
    results['HDBSCAN'] = {
        'labels': hdbscan_labels,
        'scores': evaluate_clustering(hdbscan_labels, xtr_train_dense),
    }
    plot_cluster_results(xtr_tsne, hdbscan_labels, "HDBSCAN Model")
    
    #6. OPTICS
    pca = PCA(n_components=0.95)  # 保留95%的方差
    xtr_reduced = pca.fit_transform(xtr_train_dense)
    optics = OPTICS(
        min_samples=3,
        xi=0.03,
        min_cluster_size=15,
        metric='euclidean',  # 改为欧氏距离
        algorithm='auto',    # 让sklearn自动选择最佳算法
        n_jobs=-1
    )
    optics_labels=optics.fit_predict(xtr_train_dense)
    results['OPTICS'] = {
        'labels': optics_labels,
        'scores': evaluate_clustering(optics_labels, xtr_train_dense),
    }
    plot_cluster_results(xtr_tsne, optics_labels, "OPTICS Model")
    

    # 打印评估结果表格
    print("\n聚类评估结果汇总:")
    print(f"{'算法':<20} {'Silhouette':<12} {'Calinski-Harabasz':<18} {'Davies-Bouldin':<15}")
    print("-"*70)
    for name, res in results.items():
        if res['scores'][0] is not None:
            print(f"{name:<20} {res['scores'][0]:<12.4f} {res['scores'][1]:<18.2f} {res['scores'][2]:<15.4f}")
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<18} {'N/A':<15}")
    
    # 确定最佳算法
    valid_results = {k: v for k, v in results.items() if v['scores'][0] is not None}
    if valid_results:
        best_algo = max(valid_results.items(), 
                       key=lambda x: x[1]['scores'][0] + x[1]['scores'][1]/1000 - x[1]['scores'][2])
        print("\n" + "="*50)
        print(f"最佳聚类算法: {best_algo[0]}")
        print(f"Silhouette Score: {best_algo[1]['scores'][0]:.4f}")
        print(f"Calinski-Harabasz Index: {best_algo[1]['scores'][1]:.2f}")
        print(f"Davies-Bouldin Index: {best_algo[1]['scores'][2]:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
