"""
some standard var_name style
"""


class TestClass(object):
    _p = 880

    def __init__(self):
        # _保护变量 protected
        self._var = 0
        # __私有成员__, 只有类对象自己能访问，子类也不行
        self.__var__ = 1


t = TestClass()
print(t._var)

# t._var = 10
# print(t._var)

print(t._p)

print(t.__var__)

