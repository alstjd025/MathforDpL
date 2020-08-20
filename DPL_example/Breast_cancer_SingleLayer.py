import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# 데이터 저장
x = cancer.data
y = cancer.target


# 데이터 셋을 훈련, 테스트 용으로 나눔
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.2, random_state=42)
# np.unique(y_train, return_counts=True) 데이터 셋의 비율이 잘 나뉘었는지 확인


# 단일층 신경 구현
class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        for i in range(epochs):  # epochs만큼 반복합니다
            loss = 0
            # 인덱스를 섞습니다
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:  # 모든 샘플에 대해 반복합니다
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y[i] - a)  # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err)  # 역방향 계산
                self.w -= w_grad  # 가중치 업데이트
                self.b -= b_grad  # 절편 업데이트
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다
                a = np.clip(a, 1e-10, 1 - 1e-10)
                loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
            # 에포크마다 평균 손실을 저장합니다
            self.losses.append(loss / len(y))

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        return np.array(z) > 0  # 스텝 함수 적용

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

# 모델 훈련
layer = SingleLayer()
layer.fit(x_train, y_train)
print(layer.score(x_test, y_test))


plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
