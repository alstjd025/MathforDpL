import math
import numpy as np
import random
# 평균, 표준편차
grade = [80, 78, 84, 69, 77,
         73, 88, 64, 91, 72,
         73, 62, 90, 83, 92,
         60, 76, 89, 68, 70]


print(np.std(grade)) # 넘파이로 구한 표준편차
total = 0
std = 0
for i in grade:
    total += i
    avg = total/len(grade)
print(avg) # 평균


for i in grade:
    std += (i - avg)**2
std = math.sqrt(std / len(grade))
print(std) # 표준편차


# 부체꼴 범위 안에 몇 개의 좌표쌍이 포함되는지 비율 확인
# 0~1 범위 안 소수 2개를 각 x, y

n = 1000
count = 0
for i in range(1000):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1:
        count += 1


print(count / n)
