'''
tag_preprocess:average_single_tag, weights:[1.0, 0.0, 0.0]を対象に,
異なる温度パラメータで学習したときのvalidationの類似度分布の変化を見る
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


