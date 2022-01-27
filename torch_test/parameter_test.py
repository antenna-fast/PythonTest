import torch
import torch.nn as nn
from torch.nn import Parameter


if __name__ == '__main__':
    bais = Parameter(torch.Tensor(10))  # 其实和Linear是一样的
    print('bais:', bais)

