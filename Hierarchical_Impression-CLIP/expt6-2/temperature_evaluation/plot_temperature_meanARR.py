'''
wandbから温度パラメータの初期値と最小のバリデーションのmean average retrieval rankを読み込んで,
横軸:温度パラメータの初期値, 縦軸:meanARR_val_minでプロット
tag_preprocess:average_single_tag, weights:[1.0, 0.0, 0.0]を対象
'''
import wandb
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def load_data():
    api = wandb.Api()
    entity = 'yugo-kubota-kyushu-university'
    project = 'Hierarchical_ImpressionCLIP_6-2'

    runs = api.runs(f"{entity}/{project}")
    run_ids = [run.id for run in runs]

    initial_temperature_list = []
    meanARR_val_min_list = []
    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.scan_history()
        tag_preprocess = run.config['tag_preprocess']
        weights = run.config['weights']
        print(run.name)
        if tag_preprocess=='average_single_tag' and weights=='[1.0, 0.0, 0.0]':
            initial_temperature = run.config['initial_temperature']
            initial_temperature_list.append(initial_temperature)
            meanARR_val_min = list(history)[-1]['meanARR_val_min']
            meanARR_val_min_list.append(meanARR_val_min)
    return initial_temperature_list, meanARR_val_min_list


# define constant
params = utils.get_parameters()
EXPT = params.expt

# ディレクトリの作成
SAVE_DIR = f'{EXPT}/temperature_evaluation/temperature_meanARR'
os.makedirs(SAVE_DIR, exist_ok=True)

# wandbからデータの読み込み
data_filename = f'{SAVE_DIR}/data.pkl'
if os.path.exists(data_filename):
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
        initial_temperature_list = data['initial_temperature_list']
        meanARR_val_min_list = data['meanARR_val_min_list']
else:
    initial_temperature_list, meanARR_val_min_list = load_data()
    data_dict = {'initial_temperature_list': initial_temperature_list,
                 'meanARR_val_min_list': meanARR_val_min_list}
    with open(data_filename, 'wb') as f:
        pickle.dump(data_dict, f)

# 散布図と平均の折れ線グラフを描画して保存
x = np.asarray(initial_temperature_list)
y = np.asarray(meanARR_val_min_list)
unique_x = np.unique(x)
mean_y = [y[x==xi].mean() for xi in unique_x]
plt.plot(unique_x, mean_y, color=plt.cm.tab10(1), linewidth=1, zorder=1)
plt.scatter(x, y, s=7, color='white', zorder=2, edgecolors='black', linewidths=0.5)
plt.xticks(unique_x, rotation=-90, fontsize=8)
plt.yticks(fontsize=8)
plt.savefig(f'{SAVE_DIR}/plot.png', bbox_inches='tight', dpi=300)
plt.close()