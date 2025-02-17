import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

# 加载保存的 embeddings
original_embeddings = np.load('original_embeddings.npy')
problem_embeddings = np.load('problem_embeddings.npy')

print(original_embeddings.shape)  # 查看 embeddings 的维度
print(problem_embeddings.shape)
# 合并 embeddings
embeddings = np.vstack((original_embeddings, problem_embeddings))

# 创建标签（原始数据的标签为 0，问题数据的标签为 1）
labels = np.array([1] * original_embeddings.shape[0] + [0] * problem_embeddings.shape[0])

# 使用 t-SNE 对 embeddings 进行降维
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 绘制 t-SNE 可视化图
plt.figure(figsize=(8, 6))

# 为原始嵌入 (original_embeddings) 和问题嵌入 (problem_embeddings) 设置不同的颜色
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=ListedColormap(['#88AFC9', '#F3D19C']), s=10, alpha=0.95)

# 获取 legend handles 和 labels
# handles, legend_labels = scatter.legend_elements() ["Problem-oriented", "User-oriented"]
from matplotlib.patches import Rectangle
legend_elements = [Rectangle((0, 0), 1, 0.2, color='#88AFC9'),
                   Rectangle((0, 0), 1, 0.2, color='#F3D19C')]
plt.legend(handles=legend_elements, labels=["Problem-oriented", "User-oriented"], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=15, frameon=True, fancybox=True, shadow=False)

# 设置标题和轴标签
# plt.title("t-SNE Visualization of Embeddings")
plt.tick_params(axis='both', labelsize=14)
plt.savefig("tsne_visualization.pdf", format="pdf", dpi=100, bbox_inches="tight")
plt.savefig("tsne_visualization.png", format="png", dpi=300, bbox_inches="tight")
# 显示图像
plt.show()