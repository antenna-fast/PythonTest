

# 参数列表说明
# 3.8引入
def a(x, b, /, d, e, *, c):
    # / 后面是必须指定位置的参数
    # * 后面的是必须指定参数key的参数
    print(x)
    print(b)
    print(c)
    print(d)
    print(e)
    return 0


if __name__ == '__main__':
    a(1, 2, 3, 5, c=4)
    print()
