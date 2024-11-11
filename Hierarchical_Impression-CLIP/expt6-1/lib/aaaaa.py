import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils




params = utils.get_parameters()
EXPT = params.expt

img_feature_path_train = f'{EXPT}/clustering/cluster/tag/normal/train/2.npz'
train1 = np.load(img_feature_path_train)['arr_0']

img_feature_path_train = f'{EXPT}/clustering/zzz_cluster_tag/train/normal/2.npz'
train2 = np.load(img_feature_path_train)['arr_0']

pass