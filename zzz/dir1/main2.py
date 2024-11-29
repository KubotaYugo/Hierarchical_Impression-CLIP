import sys
import os

# 親ディレクトリ（your_project）のパスを sys.path に追加
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# your_package をインポート
from utils1 import utils1_1