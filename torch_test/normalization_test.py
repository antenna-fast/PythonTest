import numpy as np

# bn
# 其实是标准化
# 计算：减均值/方差
# 效果：数据服从均值为0，方差为？的分布，有利于不同通道使用统一学习率进行训练

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])


print(np.mean(a, axis=-1))  # diff channel


def bn(in_data):
    in_mean = np.mean(in_data)
    return 0
