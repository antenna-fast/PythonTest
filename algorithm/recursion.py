"""
Author: ANTenna on 2021/11/24 8:19 下午
aliuyaohua@gmail.com

Description:

simplest recursion demo

get the result of 1+2+3+... +N
"""


# 连加  1+2+3+4+...+N
def recur_sum(N):
    if N <= 1:
        return 1
    return N + recur_sum(N - 1)


# 阶乘
def prod(N):
    if N <= 1:
        return 1
    return N * prod(N-1)


# 1, 1, 2, 3, 5, 8...
def fib(N):
    # end-condition
    if N <= 1:
        return 1
    # sub-problem
    else:
        return fib(N-1) + fib(N-2)


# 相反的顺序打印字符串
def inverse_print(in_str):
    # end-condition
    if len(in_str) == 1:
        print(in_str)
    # 递归条件
    else:
        print(in_str[-1])
        in_str = in_str[:-1]  # sub-problem
        return inverse_print(in_str)


# 青蛙上楼问题：
# 给定楼层N，每次可以跳1阶或者2阶，求到N层所有可能的路径
# 解：使用递归，把大规模的问题分成子问题： 到N，可以从N-1跳一次(如果这中间有多重路径，则*倍数) 或者从N-2跳一次
# 问题：存在大量的重复计算，解决：动态规划
def jump(N):
    # border
    if N <= 1:
        return 1
    else:  # recursion condition
        return jump(N-1) + jump(N-2)  # sub-problem


if __name__ == '__main__':
    # res = recur_sum(100)
    # res = fib(0)
    # res = inverse_print('1234')
    # res = prod(5)
    res = jump(40)
    print("res: ", res)
