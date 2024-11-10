#!/bin/bash

# BASE_PATH の定義
BASE_PATH="programs/Hierarchical_Impression-CLIP/experiment5/experiment5-2/clustering"

# Pythonスクリプトの実行
# python "${BASE_PATH}/bisecting_standardization_kmeans_image.py"
# python "${BASE_PATH}/bisecting_standardization_kmeans_impression.py"
python "${BASE_PATH}/quantitative_evaluation_standardization_img.py"
python "${BASE_PATH}/quantitative_evaluation_standardization_tag.py"
# python "${BASE_PATH}/bisecting_kmeans_image_PCA.py"
# python "${BASE_PATH}/bisecting_kmeans_image_tSNE.py"
# python "${BASE_PATH}/bisecting_kmeans_impression_PCA.py"
# python "${BASE_PATH}/bisecting_kmeans_impression_tSNE.py


# コマンド: bash programs/Hierarchical_Impression-CLIP/experiment5/experiment5-2/clustering/clustering2.sh