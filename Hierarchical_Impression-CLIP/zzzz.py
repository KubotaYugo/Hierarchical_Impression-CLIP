import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 点の数
num_points = 1000
# 球面上の赤い点を生成 (ランダム分布)
phi = np.random.uniform(0, 2 * np.pi, num_points)  # 0から2πの範囲
theta = np.random.uniform(0, np.pi, num_points)   # 0からπの範囲
x_red = np.sin(theta) * np.cos(phi)
y_red = np.sin(theta) * np.sin(phi)
z_red = np.cos(theta)
# 球面上の青い点を生成 (一部に集中)
phi_blue = np.random.uniform(0, 2 * np.pi, num_points // 10)  # 少数の点
theta_blue = np.random.uniform(0, np.pi / 4, num_points // 10)  # 上部に集中
x_blue = np.sin(theta_blue) * np.cos(phi_blue)
y_blue = np.sin(theta_blue) * np.sin(phi_blue)
z_blue = np.cos(theta_blue)
# プロット
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# 球体のガイドライン
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.sin(v), np.cos(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.cos(v), np.ones_like(u))
ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)
# 赤い点 (球面全体)
ax.scatter(x_red, y_red, z_red, c='red', s=2, label='Red Points')
# 青い点 (局所集中)
ax.scatter(x_blue, y_blue, z_blue, c='blue', s=30, label='Blue Points')
# プロットの調整
ax.set_box_aspect([1, 1, 1])  # 等方性
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()