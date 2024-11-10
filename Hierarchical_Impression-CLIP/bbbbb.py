def has_duplicates(lst):
    # リスト内のリストをタプルに変換し、セットに追加
    seen = set()
    for sublist in lst:
        # サブリストをタプルに変換してセットに追加
        sublist_tuple = tuple(sublist)
        if sublist_tuple in seen:
            return True  # 重複が見つかった場合
        seen.add(sublist_tuple)
    return False  # 重複なし

# 実行例
my_list = [['happy', 'cool', 'serif'], ['happy', 'serif'], ['happy', 'cool']]
result = has_duplicates(my_list)
print(result)  # 重複あり → True
