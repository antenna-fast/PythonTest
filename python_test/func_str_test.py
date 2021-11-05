

class My(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):  # 当打印My时，直接返回此处的str, 相当于重载()运算符
        return self.name


print(My('123'))
