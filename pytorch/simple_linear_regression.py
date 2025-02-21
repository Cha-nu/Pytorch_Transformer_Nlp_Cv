import numpy as np
from sklearn.preprocessing import StandardScaler

# 데이터 준비
x = np.array([[i] for i in range(1, 31)], dtype=np.float32)
y = np.array([[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
              [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
              [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]], dtype=np.float32)

# 데이터 정규화
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

# 파라미터 초기화
weight = np.random.randn() * 0.01  # 작은 랜덤값으로 초기화
bias = np.random.randn() * 0.01
learning_rate = 0.001  # 학습률 조정

# 학습 루프
epochs = 10000
for epoch in range(epochs):
    yh = weight * x + bias
    cost = ((y - yh) ** 2).mean()

    # 수동 경사 하강법 (flatten 추가)
    grad_w = -2 * ((y - yh).flatten() * x.flatten()).mean() # flatten() 2차원 배열의 브로드 캐스트를 방지하기 위해 1차원 배열로 변환
    grad_b = -2 * (y - yh).mean()

    weight -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    # NaN 체크
    if np.isnan(weight) or np.isnan(bias):
        print(f"NaN detected at epoch {epoch + 1}! Stopping training.")
        break

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}. Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")
