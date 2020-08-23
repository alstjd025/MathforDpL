#  훈련 : 검증 : 테스트 6 : 2 : 2 비율로 나눈 모델
#  사이킷런의 SGDClassifier 클래스를 이용하여 훈련


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

#  데이터 셋을 훈련 : 테스트 8:2로 나눔
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2,
                                                  random_state=42)
#  훈련 : 테스트 8:2로 나눈 뒤 훈련셋을 훈련 : 검증 6:2로 나눔

#  print(len(x_train), len(x_val))  비율 확인

# 모델 훈련
sgd = SGDClassifier(loss='log', random_state=42)
sgd.fit(x_train, y_train)
print(sgd.score(x_val, y_val))




