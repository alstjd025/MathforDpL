import math
import numpy as np


def var(data):
    avg = sum(data) / len(data)
    total = 0
    for i in data:
        total += (avg - i) ** 2
    return total / len(data)


def std(data):
    return math.sqrt(var(data))


a = [72, 61, 91, 31, 45]


print(sum(a) / len(a))  # 평균
print(var(a))  # 분산
print(std(a))  # 표준편차


print(np.var(a))
print(np.std(a))


c = [173, 181, 168, 175, 179]
d = [1.73, 1.81, 1.68, 1.75, 1.79]


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


print(standardize(c))
print(standardize(d))




