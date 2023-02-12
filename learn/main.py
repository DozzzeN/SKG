#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import os
from collections.abc import Iterable


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


a = 100
if a >= 0:
    print(a)
else:
    print(-a)

print([x * x for x in range(1, 11) if x % 2 == 0])
print([m + n + str(o) for m in 'ABC' for n in 'XYZ' for o in range(1, 3)])
print([d for d in os.listdir('..')])

L1 = ['Hello', 'World', 18, 'Apple', None]
print([s.lower() for s in L1 if isinstance(s, str)])


def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'


for s in fib(6):
    print(s)

print(isinstance((x for x in range(10)), Iterable))


class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))


bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
print(bart.print_score())
print(lisa.print_score())

bart.age = 8
print(bart.age)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
