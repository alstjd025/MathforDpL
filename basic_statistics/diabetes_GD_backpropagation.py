#경사하강법 역전파
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()


x = diabetes.data[:, 2]
y = diabetes.target
w = 1.0
b = 1.0

y_hat = x[0] * w + b
w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b
w_rate = (y_hat_inc - y_hat) / (w_inc - w)

err = y[0] - y_hat
print("y[0]와의 오차 : ", err)
w_new = w + w_rate * err
b_new = b + 1 * err
print(w_new, b_new)


#  두번쨰 셈플 x[1]에 대해 오차를 구하여 역전파

y_hat = x[1] * w_new + b_new
w_rate = x[1]  # (y_hat_inc - y_hat) / (w_inc - w) 계산하여 정리하면 x이므로

err = y[1] - y_hat
print("y[1]과의 오차 : ", err)
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)


#  반복문을 이용하여 역전파 과정 반복, 100번 에포크
for i in range(1, 100):
  for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print("최종값 : ", w, b)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
