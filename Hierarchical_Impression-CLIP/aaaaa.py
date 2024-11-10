import itertools

# def make_str(x, length):
#     all_combinations = itertools.product(x, repeat=length)
#     unique_combinations = [''.join(p) for p in all_combinations if len(set(p)) == len(p)]
#     return unique_combinations

# def padding_tags(s_org, max_length=10):
#     s = s_org
#     while len(s)+len(s_org)<=max_length:
#         s = s+s_org
#     all_combinations = itertools.product(s_org, repeat=max_length-len(s))
#     add = [''.join(p) for p in all_combinations if len(set(p)) == len(p)]
#     return_list = [s+a for a in add]
#     return return_list

# stt = 'ABCDEFGHIJ'
# for i in range(10):
#     s = stt[:i+1]
#     print(f"\nGenerated combinations from '{s}':")
#     print(len(padding_tags(s)))

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

def padding_tags(tags_org, max_length=10):
    tags = tags_org
    while len(tags)+len(tags_org)<=max_length:
        tags = tags+tags_org
    all_combinations = itertools.product(tags_org, repeat=max_length-len(tags))
    add = [list(p) for p in all_combinations if len(set(p)) == len(p)]
    if add!=['']:
        return_list = [tags+a for a in add]
    else:
        return_list = tags
    return return_list

tags_org = ['happy', 'serif', 'conda']
# tags_org = ['a', 'b', 'c', 'd', 'e']
for i in range(len(tags_org)):
    tags = tags_org[:i+1]
    print(f"\nGenerated combinations from '{tags}':")
    ans = padding_tags(tags)
    print(f'len={len(ans)}, {has_duplicates(ans)}')
    print(ans)