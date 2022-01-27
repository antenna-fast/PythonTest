
class A(object):
    def __init__(self, name, *height):
        # 参数表：
        # 0. 什么前缀都不加 必备参数
        # 可变长度参数：
        # 1. * 不带key的，仅仅是value ()
        # 2. ** {} 需要给定参数的{key: value}
        self.name = name
        self.height = height


class B(A):
    def __init__(self, age):
        self.age = age


class C(A):
    def __init__(self, name, age):
        super(C, self).__init__(name=name)  # 继承父类的init 同时，本类的init也有效
        self.age = age


a = A('hhh')
b = B(10)
c = C('hhh', 10)

print(a.name)
print(a.height)

# print(b.name)  # 报错 因为A的init被覆盖掉了
print(b.age)

print(c.name, c.age)
