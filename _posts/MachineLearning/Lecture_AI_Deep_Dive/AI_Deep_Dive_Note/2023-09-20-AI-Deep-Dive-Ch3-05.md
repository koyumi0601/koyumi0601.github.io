---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 05. GD vs SGD (Stochastic Gradient descent)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 05. GD vs SGD (Stochastic Gradient descent)

# 강의

- 미분을 구할 때, L (sum of e^2, 전체데이터)가 아니라 e3 하나만 보고 갈 수 있다.

![ch0305_6]({{site.url}}/images/$(filename)/ch0305_6.png)

![ch0305_5]({{site.url}}/images/$(filename)/ch0305_5.png)

![ch0305_1]({{site.url}}/images/$(filename)/ch0305_1.png)



- 데이터 개수 * 스텝이 계산량이므로, SGD는 여덟번 이동, GD는 5*4번 이동



```python
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
```



# pytorch

```python
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
```

![ch0305_2]({{site.url}}/images/$(filename)/ch0305_2.png)

- 파이토치는 자동 미분 기능, 연산 최적화, GPU 가속 등을 통해 작은 미니배치 크기에서도 경사 하강법이 더 안정적으로 수렴하게 해 준다

# tensorflow

```python
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
    # 각 스텝마다 임의의 미니배치 선택, 비복원추출 np.random.choice
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
```



![ch0305_4]({{site.url}}/images/$(filename)/ch0305_4.png)