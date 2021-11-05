# 1.1
# *args 可变数量的参数
def args_test(name, *args):
    print('name: {}'.format(name))
    # type(args) = tuple
    for i, arg in enumerate(args):
        print('args_{}: {}'.format(i, arg))


# args_test('123', '234', '456')

# 1.2
# **kwargs
# 允许你将不定长度的**键值对**, 作为参数传递给一个函数
# 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs

def kwargs_test(**kwargs):
    for key, value in kwargs.items():
        print('key: {0:<10} value:{1:<15}'.format(key, value))


# kwargs_test(name=1, age=2, height=3)


# 1.3
# 使用 *args 和 **kwargs调用函数


def kwargs_func_test(arg1, arg2, arg3):
    print('arg1: ', arg1)
    print('arg2: ', arg2)
    print('arg3: ', arg3)


# 1.3.1 使用*args  tuple里面包含了参数顺序
args = ('two', 3, 5)
# kwargs_func_test(*args)

# 1.3.2 使用**kwargs  可以使用dict指定key-value mapping
kwargs = {'arg1': 3, 'arg3': 6, 'arg2': 9}
# kwargs_func_test(**kwargs)

# 1.3.3
# 同时使用的顺序 [固定长度的参数】 [*args] [**kwargs]
# some_func(arg1, *args, **kwargs)

# When to use??
# 这还真的要看你的需求而定。
# 最常见的用例是在写函数装饰器的时候（会在另一章里讨论）
# 此外它也可以用来做猴子补丁(monkey patching)

