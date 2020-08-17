#수동적인 방법으로 가중치와 절편 조절
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()


x = diabetes.data[:, 2]
y = diabetes.target
w = 1.0
b = 1.0

w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b
y_hat = x[0] * w + b
print("y hat 증가량 : ", y_hat_inc)
print("y[0] : ", y[0])


w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print("w 증가율 : ", w_rate)   # w의 증가에 따른 y의 증가율


w_new = w + w_rate
print("w new  : ", w_new)


b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print("b 증가율 : ", b_rate)


b_new = b + 1
print("b new : ", b_new)
