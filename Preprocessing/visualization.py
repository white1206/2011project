from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42, init = 'random')
xtr_train_tsne = tsne.fit_transform(xtr_train)

# 对类别标签进行编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 创建散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(x=xtr_train_tsne[:, 0], y=xtr_train_tsne[:, 1], hue=y_train_encoded, palette='viridis')
plt.title('t-SNE Projection of Training Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Income Class')
plt.show()
