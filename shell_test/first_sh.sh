
set -e
set -x  # 显示指令 及其参数

#!/bin/zsh
# 约定的标记，它告诉系统这个脚本需要什么解释器来执行，即使用哪一种 Shell

# 内置变量
echo $(date)

# 打印到窗口
echo "Hello World !"  # echo 命令用于向窗口输出文本

# 变量名外面加不加花括号都可以 最好加上，因为可以分清边界
name_id=1
echo "name_id is "${name_id}

# 只读变量
readonly face=123
echo "face_id is "${face}

# 删除变量
# 删除后不可以再使用，不可以删除只读变量
unset face

# 字符串

# 单引号 / 双引号
# 用于字符串出现空格时
# 单引号：都是普通字符，即使是特殊字符也不具有特殊性

your_name="yaohua"
str="Hello, my name is \"${your_name}\" "
echo "${str}"  # 输出是str的内容
echo '${str}'  # 此时，输出是 ${str} 这是错误的

# int to str
a=123.456
echo name is "${a}"

# 引号
single='I am yaohua!!!'
echo "${single}"  # 双引号可以使用变量
echo '${single}'  # 单引号只能原样输出, 不能出现一个单独的引号

# 拼接
echo 'my ''name'

# 字符串长度
echo 'len_single: '${#single}

# 提取子字符串
echo "${single:1:4}"
