import numpy as np

#  각 행렬의 역행렬 구함
A = np.array([[3, -2], [4, 5]])
B = np.array([[1, -3, 4], [-2, 5, 3], [2, -1, 0]])
C = np.array([[-1, 5, 2, -3], [0, 3, -1, 4], [2, -3, 0, -5], [-4, 2, 3, 1]])


print(np.linalg.inv(A))
print(np.linalg.inv(B))
print(np.linalg.inv(C))


#  각 연립방정식의 해를 구함
D = np.array([[2, 5, -4], [3, -2, 6], [-1, 3, -2]])
D_ = np.array([9, 9, 4])
E = np.array([[-1, 3, 5, 2], [4, -6, 3, -1], [3, 3, -4, 3], [2, -1, 2, -4]])
E_ = np.array([8, -32, 26, -25])


print(np.linalg.inv(D) @ D_)
print(np.linalg.inv(E) @ E_)

