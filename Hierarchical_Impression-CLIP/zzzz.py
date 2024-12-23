import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# ランダムなデータセットを作成
# np.random.seed(0)
# data = np.random.rand(5, 2)  # 5つのサンプル、2次元データ
data = np.asarray([[0.1], [0.2], [0.5], [0.6], [0.9]])

# 階層クラスタリングを実行
Z = linkage(data, method='ward')
for i in range(Z.shape[0]):
    Z[i][2] = i+1

# リンク行列を表示
print(Z)

dendrogram(Z)
plt.show()