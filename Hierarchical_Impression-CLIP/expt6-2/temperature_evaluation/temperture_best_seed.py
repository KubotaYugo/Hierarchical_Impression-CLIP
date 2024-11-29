'''
tag_preprocess:average_single_tag, weights:[1.0, 0.0, 0.0]を対象に,
各温度パラメータでmeanARR_val_minが最も小さいときの乱数のシード値を取得
'''
import wandb
import csv
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt

# wandbの読み込み
api = wandb.Api()
entity = 'yugo-kubota-kyushu-university'
project = 'Hierarchical_ImpressionCLIP_6-2'
runs = api.runs(f"{entity}/{project}")
run_ids = [run.id for run in runs]

# wandbからデータの取得
initial_temperature_list = []
meanARR_val_min_list = []
random_seed_list = []
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.scan_history()
    tag_preprocess = run.config['tag_preprocess']
    weights = run.config['weights']
    print(run.name)
    if tag_preprocess=='average_single_tag' and weights=='[1.0, 0.0, 0.0]':
        # initial_temperature
        initial_temperature = run.config['initial_temperature']
        initial_temperature_list.append(initial_temperature)
        # meanARR_val_min
        meanARR_val_min = list(history)[-1]['meanARR_val_min']
        meanARR_val_min_list.append(meanARR_val_min)
        # random_seed
        random_seed = run.config['random_seed']
        random_seed_list.append(random_seed)

# numpyに変換
initial_temperature_list = np.asarray(initial_temperature_list)
meanARR_val_min_list = np.asarray(meanARR_val_min_list)
random_seed_list = np.asarray(random_seed_list)

# 各温度パラメータでmeanARR_val_minを最小にするrandom_seedの取得
best_random_seed_dict = {}
unique_initial_temperature = np.unique(initial_temperature_list)
for initial_temperature in unique_initial_temperature:
    mARR = meanARR_val_min_list[initial_temperature_list==initial_temperature]
    rs = random_seed_list[initial_temperature_list==initial_temperature]
    best_random_seed = random_seed_list[np.argmin(mARR)]
    best_random_seed_dict[initial_temperature] = best_random_seed
print(best_random_seed_dict)

# csvで保存
save_list = [list(best_random_seed_dict.keys()), list(best_random_seed_dict.values())]
filename = f'{EXPT}/temperature_evaluation/best_seed.csv'
utils.save_list_to_csv(save_list, filename)