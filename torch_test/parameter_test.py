import torch
import torch.nn as nn
from torch.nn import Parameter


if __name__ == '__main__':
    bais = Parameter(torch.Tensor(10))
    print('bais:', bais)

