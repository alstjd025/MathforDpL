import numpy as np
import matplotlib.pylab as plt


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative 도함수
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Graph
t = np.arange(-10, 10, 0.01)
print(sigmoid_derivative(0))
plt.plot(t, sigmoid(t), label="Sigmoid(x)")
plt.plot(t, sigmoid_derivative(t), label="Sigmoid_der(x)")


plt.ylim(0, 1)
plt.legend()
plt.show()
