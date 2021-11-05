# 利用好调试，能大大提高你捕捉代码Bug的
# 大部分新人忽略了Python debugger(pdb)的重要性
# 在这个章节我只会告诉你一些重要的命令，你可以从官方文档中学习到更多

# 1. 从命令行运行
# python -m pdb test.py

# 2. 从脚本内部运行 [在脚本内部设置断点]
import pdb


def make_bread():
    pdb.set_trace()
    return "I don't have time "


print(make_bread())

# debugger命令

