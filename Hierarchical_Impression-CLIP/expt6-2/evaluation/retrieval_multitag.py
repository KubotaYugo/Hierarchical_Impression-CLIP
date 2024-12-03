'''
2つのタグ(全組み合わせ)での検索結果でPrecision@K, Average_precisionを求める
'''
# 追記: 各ペア毎の評価指標を考えると，数が膨大なので，mean Average Precisionだけでいいかも
# MyFontsの論文に載っていたようにデータセットからランダムに3つ抽出というようにして評価してもいいかも
# For each font in MyFontstest, we randomly generate 3 subsets of its ground-truth tags as 3 multi-tag queries. 
# The query size ranges from 2 to 5 tags. After filter out repeating queries, this query set contains 5,466 multi-tag queries.