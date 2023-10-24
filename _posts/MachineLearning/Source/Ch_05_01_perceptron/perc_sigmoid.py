import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 데이터 생성
x = np.linspace(0, 200, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
Z = sigmoid(Y - X - 1)  # Y = X + 1을 경계로 시그모이드 함수 적용

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Need to Lose Weight')
ax.set_title('3D Plot of Weight vs Height with Sigmoid Mapping')

# 각 축의 범위 설정
ax.set_xlim([0, 200])
ax.set_ylim([0, 100])
ax.set_zlim([0, 1])

plt.show()