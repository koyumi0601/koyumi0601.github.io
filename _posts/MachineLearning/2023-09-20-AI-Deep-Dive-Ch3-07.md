---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 07. Moment vs RMS Prop"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 07. Moment vs RMS Prop

*비등방함수의 수렴(지그재그)을 완화하기 위한 최적화 기법들*

<br>

- (좌) mini-batch SGD (우) momentum

![ch0307_1]({{site.url}}/images/$(filename)/ch0307_1.png)



- 원이 아니라, 타원이라면, gradient descent를 이용해서 이동할 때, 꼭짓점으로 바로 가는 게 아니라, 등고선에 수직한 방향으로 지그재그로 움직인다.

- 타원이 찌그러져 있을 수록 더 왔다갔다 한다

- 모멘텀은 반면, 관성을 주어 움직인다. 이전의 움직인 정보를 더해서(그라디언트를 누적해서) 움직인다. 방향이 급격하게 틀어지는 것을 완화해 준다.

- 좌우로 움직이는 것은 서로 상쇄되나, 앞으로 가는 것은 가속화된다.

- 다 와서는 목표지점을 지나치기도 한다.
  - ex. 치타가 방향전환하는 것





#### 추가조사

##### 계산수식

![ch0307_4]({{site.url}}/images/$(filename)/ch0307_4.png)

- moving average라고 이해하면 될 것 같다
- learning rate의 dynamic한 조정이라고 볼 수 있다.



<br><br>

<br>

# RMS prop. (Root Mean Square Propagation)

#### 강의 내용

![ch0307_2]({{site.url}}/images/$(filename)/ch0307_2.png)

### 추가조사

- 제안: 

  - Geoffrey Hinton의 강의([강의록](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))에서 처음 제안됨. 

- 핵심 아이디어

  - 각 파라미터에 대해 학습률을 개별적으로 조정
  - 그래디언트가 작은 파라미터는 더 큰 학습률을 가지며, 그래디언트가 큰 파라미터는 더 작은 학습률을 가짐

- 수식

  ![ch0307_3]({{site.url}}/images/$(filename)/ch0307_3.png)

- 효과성

  - 이러한 동적 학습률 조정은 비등방성(anisotropic) 비용 함수에서의 최적화에서 성능 향상.

- 예시

  - 사람이 경사가 가파를 때는 작은 걸음을, 경사가 완만할 때는 큰 걸음을 걷는 것과 유사함.

- 논문x. 수학적 증명x 



<br>

<br>

#### Momentum, RMSProp 차이점

- **Momentum**은 이전 그래디언트의 방향을 유지하여 지그재그 움직임을 줄이고, 최적점에 더 빠르게 도달
- **RMSProp**은 각 파라미터에 대해 적응적으로 학습률을 조정하여, 다양한 스케일의 파라미터를 효과적으로 최적화







> **최적화 알고리즘**
>
> GD, SGD, Minibatch SGD, Momentum, RMS Prop. **Adam**, Adagrad, Adadelta, FTRL(Follow The Regularized Leader), L-BFGS(Limited-memory Broyden-Fletcher-Goldfarb-Shanno)

#### 주로 사용될 때

##### Adam:

- Adam은 Momentum과 RMSprop의 아이디어를 결합하여 개발되었습니다.
- Adam은 학습률 스케줄링을 내장하고 있어, 사용자가 학습률을 조정할 필요가 적습니다.
- 다양한 문제와 데이터 유형에 대해 잘 작동하는 것으로 알려져 있습니다.

**SGD (Stochastic Gradient Descent):**

- 기본적이고 간단한 최적화 알고리즘으로, 경우에 따라 잘 작동할 수 있습니다.

- **RMSprop:**
  - 비등방성 함수에서의 성능이 좋습니다.
- **Adagrad, Adadelta:**
  - 특정 문제에서 잘 작동할 수 있습니다.
- **L-BFGS:**
  - 주로 볼록 최적화 문제에 사용됩니다.



#### 코드

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # pytorch
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # tensorflow
```





### 코드

![ch0307_5]({{site.url}}/images/$(filename)/ch0307_5.png)

*연한 색: 초기 위치, 진한 색: 나중 위치, 빨간 색: 최종 위치*

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# 초기값 및 하이퍼파라미터 설정
w1_init = 1.0
w2_init = 1.0
learning_rate = 0.1
num_steps = 50

# Optimizer 클래스 정의
class SGDOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, w1, w2, grad_w1, grad_w2):
        w1 -= self.learning_rate * grad_w1
        w2 -= self.learning_rate * grad_w2
        return w1, w2

class MomentumOptimizer:
    def __init__(self, learning_rate, alpha=0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.v_w1 = 0
        self.v_w2 = 0
    
    def update(self, w1, w2, grad_w1, grad_w2):
        self.v_w1 = self.alpha * self.v_w1 + self.learning_rate * grad_w1
        self.v_w2 = self.alpha * self.v_w2 + self.learning_rate * grad_w2
        w1 -= self.v_w1
        w2 -= self.v_w2
        return w1, w2

class RMSPropOptimizer:
    def __init__(self, learning_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.s_w1 = 0
        self.s_w2 = 0
    
    def update(self, w1, w2, grad_w1, grad_w2):
        self.s_w1 = self.beta * self.s_w1 + (1 - self.beta) * grad_w1 ** 2
        self.s_w2 = self.beta * self.s_w2 + (1 - self.beta) * grad_w2 ** 2
        w1 -= self.learning_rate * grad_w1 / (np.sqrt(self.s_w1) + 1e-10)
        w2 -= self.learning_rate * grad_w2 / (np.sqrt(self.s_w2) + 1e-10)
        return w1, w2

# 손실 함수와 그래디언트 정의
def f(w1, w2):
    return w1**2 + w2**2

def gradient_w1(w1):
    return 2 *  w1

def gradient_w2(w2):
    return 2 * w2

# Optimizer 인스턴스 생성
sgd_optimizer = SGDOptimizer(learning_rate)
momentum_optimizer = MomentumOptimizer(learning_rate)
rmsprop_optimizer = RMSPropOptimizer(learning_rate)

# 경로 저장을 위한 리스트 초기화
sgd_path = [(w1_init, w2_init)]
momentum_path = [(w1_init, w2_init)]
rmsprop_path = [(w1_init, w2_init)]

# 경사 하강법 수행
w1_sgd, w2_sgd = w1_init, w2_init
w1_momentum, w2_momentum = w1_init, w2_init
w1_rmsprop, w2_rmsprop = w1_init, w2_init

for step in range(num_steps):

    grad_sgd_w1 = gradient_w1(w1_sgd)
    grad_sgd_w2 = gradient_w2(w2_sgd)

    grad_momentum_w1 = gradient_w1(w1_momentum)
    grad_momentum_w2 = gradient_w2(w2_momentum)

    grad_rmsprop_w1 = gradient_w1(w1_rmsprop)
    grad_rmsprop_w2 = gradient_w2(w2_rmsprop)

    w1_sgd, w2_sgd = sgd_optimizer.update(w1_sgd, w2_sgd, grad_sgd_w1, grad_sgd_w2)
    sgd_path.append((w1_sgd, w2_sgd))
    
    w1_momentum, w2_momentum = momentum_optimizer.update(w1_momentum, w2_momentum, grad_momentum_w1, grad_momentum_w2)
    momentum_path.append((w1_momentum, w2_momentum))
    
    w1_rmsprop, w2_rmsprop = rmsprop_optimizer.update(w1_rmsprop, w2_rmsprop, grad_rmsprop_w1, grad_rmsprop_w2)
    rmsprop_path.append((w1_rmsprop, w2_rmsprop))

# 그래프 생성 및 플로팅
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(15, 5))

def plot_path(ax, path, title, color):
    ax.set_title(title)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    x, y = zip(*path)
    num_points = len(path)
    colors = [mcolors.to_rgba(color, alpha=i/num_points) for i in range(num_points)]  # 그라데이션 적용
    ax.scatter(x, y, [f(w1, w2) for w1, w2 in path], c=colors, marker='o', s=30)
    ax.scatter(x[-1], y[-1], [f(w1, w2) for w1, w2 in [path[-1]]], c='red', marker='o', s=50)  # 최종 도착 지점을 다른 색으로 표시
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Loss Function')

ax1 = fig.add_subplot(131, projection='3d')
plot_path(ax1, sgd_path, 'SGD for $f(w1, w2) = w1^2 + w2^2$', 'blue')

ax2 = fig.add_subplot(132, projection='3d')
plot_path(ax2, momentum_path, 'Momentum for $f(w1, w2) = w1^2 + w2^2$', 'blue')

ax3 = fig.add_subplot(133, projection='3d')
plot_path(ax3, rmsprop_path, 'RMSProp for $f(w1, w2) = w1^2 + w2^2$', 'blue')

plt.show()
```

