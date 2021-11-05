from collections import deque

# 双端队列
a = deque(maxlen=10)

a.appendleft(1)
a.append(2)
a.appendleft(3)

print(a)
