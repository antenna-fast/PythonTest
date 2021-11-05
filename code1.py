import numpy as np


array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])


def find(target_num):
    for i, value in enumerate(array[:, 0]):
        if target_num > value:
            if target_num < array[i][-1]:
                col_space = array[i - 1]
                for j in col_space:
                    if j == target_num:
                        return 1
                    else:
                        return 0
            else:
                continue


if __name__ == '__main__':
    # target_num = 8
    # res = find(target_num)
    # print(res)

    k = 3
    n = 10
    vec = list(range(10))

    res_vec = []
    for i in vec:

        if len(res_vec) == 0:
            res_vec.append(i)

        if i > max(res_vec):
            res_vec.append(i)

            if len(res_vec) > k:
                min_value = min(res_vec)
                res_vec.remove(min_value)

    print(res_vec)
