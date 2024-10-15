import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# サンプルデータを生成 (ここではランダムデータを使用)
data = np.random.rand(100, 50)  # 100サンプル、50次元のデータ
colors = np.random.rand(100) * 200  # 色を決定するためのリスト（0-200の範囲）

# t-SNEを実行
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(data)

# カラースケールの設定
vmin = 0     # カラースケールの最小値
vmax = 200   # カラースケールの最大値
center = 190 # カラーバーの中央の値

# 散布図を作成
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='coolwarm', vmin=vmin, vmax=vmax)

# カラーバーを表示
cbar = plt.colorbar(label='Color Scale (0-200)')

# カラーバーの中央を変更
cbar.set_ticks([vmin, center, vmax])  # 中央を指定
cbar.set_ticklabels([str(vmin), str(center), str(vmax)])  # タイプラベルを設定

# グラフのラベルやタイトル
plt.title("t-SNE with Custom Center Colorbar")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# グラフを表示
plt.show()
