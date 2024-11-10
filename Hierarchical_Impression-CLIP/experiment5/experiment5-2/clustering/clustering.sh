#!/bin/bash

# BASE_PATH の定義
BASE_PATH="programs/Hierarchical_Impression-CLIP/experiment5/experiment5-2/clustering"

# Pythonスクリプトの実行
python "${BASE_PATH}/bisecting_kmeans_image.py"
python "${BASE_PATH}/bisecting_kmeans_impression.py"
# python "${BASE_PATH}/bisecting_kmeans_image_PCA.py"
# python "${BASE_PATH}/bisecting_kmeans_image_tSNE.py"
# python "${BASE_PATH}/bisecting_kmeans_impression_PCA.py"
# python "${BASE_PATH}/bisecting_kmeans_impression_tSNE.py"

# コマンド: bash programs/Hierarchical_Impression-CLIP/experiment5/experiment5-2/clustering/clustering.sh