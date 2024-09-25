import matplotlib.pyplot as plt
import numpy as np

# データの作成
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.sinh(x)

# Figure と最初のプロットを作成
fig, ax1 = plt.subplots()

# 最初の y 軸 (ax1)
ax1.plot(x, y1, 'g-', label='sin(x)')
ax1.set_ylabel('sin(x)', color='g')

# 2つ目の y 軸 (ax2)
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='cos(x)')
ax2.set_ylabel('cos(x)', color='b')

# 3つ目の y 軸 (ax3) を作成
ax3 = ax1.twinx()

# ax3 を右側に少しずらす
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(x, y3, 'r-', label='tan(x)')
ax3.set_ylabel('tan(x)', color='r')

# 4つ目の y 軸 (ax4) を作成
ax4 = ax1.twinx()

# ax4 をさらに右側にずらす
ax4.spines['right'].set_position(('outward', 120))
ax4.plot(x, y4, 'm-', label='sinh(x)')
ax4.set_ylabel('sinh(x)', color='m')

# グラフの表示
fig.tight_layout()
plt.show()
