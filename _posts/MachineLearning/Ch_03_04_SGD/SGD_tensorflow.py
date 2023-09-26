import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 초기값 및 하이퍼파라미터 설정
a_init = 1.0
b_init = 1.0
learning_rate = 0.1
num_steps = 50
mini_batch_size = 10  # 미니배치 크기 설정

# 데이터 생성 (임의의 데이터)
data = np.random.rand(100, 2)
x_data = data[:, 0]
y_data = data[:, 1]

# 경사 하강법 수행
a = tf.Variable(a_init, dtype=tf.float32)
b = tf.Variable(b_init, dtype=tf.float32)
a_history = [a_init]
b_history = [b_init]

for step in range(num_steps):
    # 각 스텝마다 임의의 미니배치 선택
    indices = np.random.choice(len(x_data), mini_batch_size, replace=False)
    x_mini_batch = x_data[indices]
    y_mini_batch = y_data[indices]

    # 손실 함수 정의 (미니배치에 대한 손실)
    with tf.GradientTape(persistent=True) as tape:
        loss = tf.reduce_mean((a * x_mini_batch + b - y_mini_batch)**2)

    grad_a = tape.gradient(loss, a)
    grad_b = tape.gradient(loss, b)

    # 경사 하강법 업데이트
    a.assign_sub(learning_rate * grad_a)
    b.assign_sub(learning_rate * grad_b)

    # 현재 위치 기록
    a_history.append(a.numpy())
    b_history.append(b.numpy())

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