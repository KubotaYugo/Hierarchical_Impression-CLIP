import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
SAVE_FOLDER = "Hierarchical-CLIP/results"
df = pd.read_csv('Hierarchical-CLIP/results/result_epoch.csv')

# 1列目を横軸に設定
x = df.iloc[:, 0]

# 2, 3, 4列目を縦軸に設定
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]
y3 = df.iloc[:, 3]

# グラフ1
plt.plot(x, y1)
plt.xlabel('Epoch')
plt.ylabel('Loss_img')
plt.savefig(f'{SAVE_FOLDER}/Loss_img.png', bbox_inches='tight', dpi=300)
plt.close()


# グラフ2
plt.plot(x, y2)
plt.xlabel('Epoch')
plt.ylabel('Loss_tag')
plt.savefig(f'{SAVE_FOLDER}/Loss_tag.png', bbox_inches='tight', dpi=300)
plt.close()

# グラフ3
plt.plot(x, y3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f'{SAVE_FOLDER}/Loss.png', bbox_inches='tight', dpi=300)
plt.close()