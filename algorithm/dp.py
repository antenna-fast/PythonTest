"""
Author: ANTenna on 2021/11/24 2:52 下午
aliuyaohua@gmail.com

Description:
dynamic programming:
solve complex problem by divide into sample sub-problem

step:
1. define the sub-problem
2. write down the recurrence related to sub-problem
3. solve the base problem

最简单的例子：
设定N=6
给出1，3，4 求所有sum=N的路径
"""


def jump(N):
    res = 0
    if N == 1:  # when N=1, there are 1 path only
        return 1
    if N == 2:  # when N=2, there are two different path
        return 1
    for i in range(3, N+1):
        res = res + jump(i)
    return res


if __name__ == '__main__':
    print()
    # print(res)


