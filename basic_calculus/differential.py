import numpy as np
# 미분


# y=x^2
def f(x):
    return x ** 2


# 극한
def df(x, h):
    return (f(x + h) - f(x)) / h


# lim h -> 0
for h in [1, 1e-1, 1e-2, 1e-3]:
    print([h, df(0, h), df(1, h)])

