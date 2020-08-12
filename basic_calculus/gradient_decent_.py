# Gradient decent example 1
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x, y):
    return x * x + y * y


def dfx(x, y):
    return 2 * x


def dfy(x, y):
    return 2 * y


learning_rate = 0.1  # 학습률
x, y = 10, 8  # 시작점 starting point
print(f(x, y))


for i in range(0, 30):
    x += - learning_rate * dfx(x, y)
    y += - learning_rate * dfy(x, y)
    print(f(x, y))


# show x^2 + y^2 Graph
a = np.arange(-10, 10, 1)
b = np.arange(-10, 10, 1)


temp = np.zeros((len(a), len(b)))
for i in range(20):
    for j in range(20):
        temp[i, j] = f(a[i], b[j])


aa, bb = np.meshgrid(a, b)


plt.figure(figsize=(5, 3.5))
ax = plt.subplot(1, 1, 1, projection='3d')
ax.plot_wireframe(aa, bb, temp, rstride=1, cstride=1)
plt.show()
