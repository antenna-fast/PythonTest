import numpy as np

"""
轨迹预测模型
"""


def predict():

    return 0


if __name__ == '__main__':

    # 给定上一时刻的状态【速度和位置】，预测下一时刻状态
    delta_t = 1  # 系统变量
    p_last = 10  # 状态变量
    v_last = 5  # 状态变量
    a = 1  # 状态变量
    x_last = np.array([[p_last],
                       [v_last]])
    # 1. 状态预测矩阵
    F = np.array([[1, delta_t],
                  [0, 1]])
    # 2. 控制矩阵 计算外部控制对于系统状态的影响
    B = np.array([[delta_t**2 / 2], [delta_t]])
    x_state = np.dot(F, x_last) + B * a  # 状态方程

    # 测量方程
    H = np.array([])
