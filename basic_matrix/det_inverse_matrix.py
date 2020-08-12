import numpy as np


A = np.array([[2, 5], [1, 3]])
print(np.linalg.inv(A))  # 역행렬


B = np.array([[1, 4, 2], [3, -1, -2], [-3, 1 , 3]])
print(np.linalg.det(B))  # 행렬식


C = np.array([[2, 3], [-1, 4]])
B = np.array([7, 2])


print(np.linalg.inv(A) @ B)  # 좌측 행렬의 역행렬을 우측 행렬에 곱하여 해를 구함




