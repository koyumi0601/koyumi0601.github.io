import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 초기값 및 하이퍼파라미터 설정
a_init = 1
b_init = 1
learning_rate = 0.1
num_steps = 50

# 경사 하강법 수행
a = a_init
b = b_init
a_history = [a]
b_history = [b]

def gradient_a(x, y):
    return 2 * x

def gradient_b(x, y):
    return 2 * y

for step in range(num_steps):
    # 무작위 데이터 포인트 선택 (SGD)
    random_x = np.random.rand()
    random_y = np.random.rand()

    grad_a = gradient_a(random_x, random_y)
    grad_b = gradient_b(random_x, random_y)

    # 경사 하강법 업데이트
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b

    # 현재 위치 기록
    a_history.append(a)
    b_history.append(b)

# 함수 정의
def f(x, y):
    return x**2 + y**2

# 그래프 생성
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 경사 하강법 스텝 별 위치 표시
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter(a_history, b_history, [f(a, b) for a, b in zip(a_history, b_history)], c='red', marker='o', label='Steps', s=30)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Function Value')
ax.set_title('Stochastic Gradient Descent for $x^2 + y^2$')
ax.legend()
plt.show()