original_list = [2, 2, 1, 4, 5, 3, 2, 5, 4]
unique_numbers = {}
new_list = []

# ユニークな数字に新しい番号を割り当て
current_number = 0
for num in original_list:
    if num not in unique_numbers:
        unique_numbers[num] = current_number
        current_number += 1
    new_list.append(unique_numbers[num])

print(new_list)  # 出力: [2, 2, 1, 4, 5, 3] の変換結果を表示