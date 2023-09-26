import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 입력 데이터 (키, 몸무게)
data = np.array([[45, 160], [50, 162], [55, 165], [63, 170], [75, 190]], dtype=np.float32)
x_train = data[:, 0]  # 키
print(f"x_train: ", x_train)
y_train = data[:, 1]  # 몸무게
print(f"y_train: ", y_train)


# 텐서로 변환
x_train = torch.from_numpy(x_train).view(-1, 1)  # 1차원 입력 데이터를 2차원으로 변환
print(f"x_train(tensor): ", x_train)

y_train = torch.from_numpy(y_train).view(-1, 1)
print(f"y_train(tensor): ", y_train)

# 선형 회귀 모델 정의 (1차원 입력, 1차원 출력)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 입력 차원: 1, 출력 차원: 1, fully connected layer

    def forward(self, x): # 보통 활성화 함수는 forward에 정의함. linear regression의 경우 활성화함수를 정의하지 않거나 항등함수를 사용
        # return F.identity(self.linear(x))
        return self.linear(x)

model = LinearRegressionModel()

# 손실 함수와 최적화기 정의
criterion = nn.MSELoss()  # 평균 제곱 오차
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 확률적 경사 하강법

# 모델 학습
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward Pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward Pass 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 학습된 모델을 시각화하기 위해 x값 생성
x_values = torch.arange(40, 80, 1).view(-1, 1).float()

# 학습된 모델로 예측
predicted_y = model(x_values)

# 결과 시각화
plt.scatter(x_train, y_train, label='Original data')
plt.plot(x_values, predicted_y.detach().numpy(), 'r-', label='Fitted line')
plt.legend()
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Linear Regression')
plt.show()

# 학습된 모델의 가중치와 편향 출력
print('학습된 모델의 가중치 (Weight):', model.linear.weight.item())
print('학습된 모델의 편향 (Bias):', model.linear.bias.item())