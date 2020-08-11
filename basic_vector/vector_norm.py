import numpy as np
import math


a = np.array([3, 2])
b = np.array([1, 4])


print(a + b)
print(a - b)
print(np.linalg.norm(a))
print(np.linalg.norm(b))


def norm(x):
  return math.sqrt(sum([i ** 2 for i in x]))


print(norm(a))
print(norm(b))
