# 用于字符串分割
# https://linux.cn/article-13177-1.html

test_str="123.txt"
AWK -F. echo ${test_str}
