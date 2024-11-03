import torch

x = torch.tensor([3, 1, 2, 3, 4, 2, 1])
unique, inverse = torch.unique(x, sorted=False, return_inverse=True)
print("unique:", unique)          # 出力: tensor([3, 1, 2, 4])
print("inverse:", inverse)        # 出力: tensor([0, 1, 2, 0, 3, 2, 1])
pass