import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_filterwise_normalized_vector(model):
    vec = []
    for param in model.parameters():
        if len(param.size()) == 2:  # 가중치가 있는 층을 확인
            norm = torch.norm(param.data, p=2, dim=1)  # L2 노름 계산
            normalized_vec = torch.randn(param.size()) * norm.unsqueeze(1)
            vec.append(normalized_vec.view(-1))
        else:
            vec.append(torch.randn(param.size()).view(-1))
    return torch.cat(vec)

def plot_loss_landscape(model, criterion, x, y, steps=100, alpha_range=(-1, 1), beta_range=(-1, 1)):
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], steps)
    beta_values = np.linspace(beta_range[0], beta_range[1], steps)
    loss_surface = np.zeros((steps, steps))

    original_params = [param.data.clone() for param in model.parameters()]
    delta = get_filterwise_normalized_vector(model)
    eta = get_filterwise_normalized_vector(model)

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # 모델 파라미터 업데이트
            for k, param in enumerate(model.parameters()):
                param.data = original_params[k] + alpha * delta[k] + beta * eta[k]
            
            # 손실 계산
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_surface[i][j] = loss.item()

            # 원래 파라미터로 복원
            for k, param in enumerate(model.parameters()):
                param.data = original_params[k]

    plt.contourf(alpha_values, beta_values, loss_surface, levels=50)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('Loss Landscape')
    plt.colorbar()
    plt.show()

def plot_loss_landscape_3d(model, criterion, x, y, steps=100, alpha_range=(-1, 1), beta_range=(-1, 1)):
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], steps)
    beta_values = np.linspace(beta_range[0], beta_range[1], steps)
    loss_surface = np.zeros((steps, steps))

    original_params = [param.data.clone() for param in model.parameters()]
    delta = get_filterwise_normalized_vector(model)
    eta = get_filterwise_normalized_vector(model)

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # 모델 파라미터 업데이트
            for k, param in enumerate(model.parameters()):
                param.data = original_params[k] + alpha * delta[k] + beta * eta[k]
            
            # 손실 계산
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_surface[i][j] = loss.item()

            # 원래 파라미터로 복원
            for k, param in enumerate(model.parameters()):
                param.data = original_params[k]

    # 3D 플롯 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha_values, beta_values)
    ax.plot_surface(X, Y, loss_surface, cmap='viridis')

    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Landscape')

    plt.show()

# # 모델, 손실 함수, 데이터 정의
# model = SimpleNet()
# criterion = torch.nn.CrossEntropyLoss()
# x = torch.randn(100, 10)  # 임의의 입력 데이터
# y = torch.randint(0, 2, (100,))  # 임의의 타겟 레이블

# # 손실 경관 시각화
# plot_loss_landscape(model, criterion, x, y)

# 3D
# 모델, 손실 함수, 데이터 정의
model = SimpleNet()
criterion = torch.nn.CrossEntropyLoss()
x = torch.randn(100, 10)  # 임의의 입력 데이터
y = torch.randint(0, 2, (100,))  # 임의의 타겟 레이블

# 손실 경관 3D 시각화
plot_loss_landscape_3d(model, criterion, x, y)