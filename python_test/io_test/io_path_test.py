import numpy as np
import os

f_path = 'test.txt'
# f = open(f_path, 'w')
f = open(f_path, 'a')  # a 是以追加的形式写入

f.write('Hello, World!\n')

for i in range(10):
    f.write('my name is ' + str(i) + '\n')

f.close()

if not os.path.exists('tttt'):
    os.mkdir('tttt')

# os.mkdir('/asd')

# 多级文件夹  OK
os.makedirs('123/456/789')

# 连续写入数据

f_path = 'test.txt'
f = open(f_path, 'w')


# 保存字典

