#!/bin/bash

# BASE_PATH の定義
BASE_PATH="programs/Hierarchical_Impression-CLIP/experiment5/experiment5-1/evaluation"

# Pythonスクリプトの実行
# python "${BASE_PATH}/save_features.py"
# python "${BASE_PATH}/quantitative_evaluation.py"
# python "${BASE_PATH}/PCA.py"
# python "${BASE_PATH}/similarity_hist_.py"
# python "${BASE_PATH}/similarity_hist_hierarchical.py"
# python "${BASE_PATH}/rank_matrix.py"
# python "${BASE_PATH}/retrieval_rank_hist.py"
# python "${BASE_PATH}/comp_similarity_embedding.py"
# python "${BASE_PATH}/comp_similarity_embedding_ratio.py"
# python "${BASE_PATH}/similarity_matrix.py"
python "${BASE_PATH}/tSNE.py"
# python "${BASE_PATH}/tSNE_tag_embedding.py"
# python "${BASE_PATH}/tSNE_with_number_of_tags.py"
# python "${BASE_PATH}/tSNE_withRR.py"
# python "${BASE_PATH}/retrieve.py"
# python "${BASE_PATH}/.py"
# python "${BASE_PATH}/.py"


# コマンド: bash programs/Hierarchical_Impression-CLIP/experiment5/experiment5-1/evaluation/evaluate.sh