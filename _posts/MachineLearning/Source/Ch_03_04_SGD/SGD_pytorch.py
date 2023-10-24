import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 초기값 및 하이퍼파라미터 설정
a_init = 1.0
b_init = 1.0
learning_rate = 0.1
num_steps = 50

# 파라미터를 텐서로 정의
a = torch.tensor(a_init, requires_grad=True)
b = torch.tensor(b_init, requires_grad=True)

# SGD 옵티마이저 설정
optimizer = optim.SGD([a, b], lr=learning_rate)

# 손실 함수 정의 (여기에서는 예제 함수인 x^2 + y^2)
def loss_function(x, y):
    return x**2 + y**2

# 저장할 리스트 초기화
a_history = [a_init]
b_history = [b_init]
loss_history = [loss_function(a, b).item()]

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 경사 하강법 수행
for step in range(num_steps):
    # 손실 계산
    loss = loss_function(a, b)
    
    # 그래디언트 초기화
    optimizer.zero_grad()
    
    # 그래디언트 계산
    loss.backward()
    
    # 가중치 업데이트
    optimizer.step()
    
    # 현재 위치 기록
    a_history.append(a.item())
    b_history.append(b.item())
    loss_history.append(loss.item())

    # 현재 위치 출력
    print(f"Step {step+1}: a = {a.item()}, b = {b.item()}, Loss = {loss.item()}")

# 3D 그래프 플롯
ax.plot(a_history, b_history, loss_history, marker='o', linestyle='-', label='Steps')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Loss Value')
ax.set_title('3D Plot of Gradient Descent for $x^2 + y^2$')
ax.legend()

plt.show()