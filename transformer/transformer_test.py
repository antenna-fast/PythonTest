import numpy as np
import torch
import torch.nn as nn


def transformer(input_x):
    input_x_T = input_x.reshape(len(input_x), -1)
    # W = np.dot(input_x_T, input_x)
    W = np.dot(input_x, input_x_T)
    output = np.dot(W, input_x)
    output = torch.softmax(output, dim=-1)
    return output


a = torch.tensor([1, 2, 3, 4])

res = transformer(a)
print('res is ', res)
