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

- 속도
  - 이전 한 시간 스텝에서의 weight(거리) 변화량
- 모멘텀 계수
  - 이전 속도의 영향정도
  - 보통 0.9로 설정한다고 되어 있으나, 아래의 실험에서는 overshooting 발생하여 0.1로 낮추었다. 튜닝해야하는 하이퍼파라미터이다.





<br><br><br>





# RMS prop. (Root Mean Square Propagation)

#### 강의 내용

![ch0307_2]({{site.url}}/images/$(filename)/ch0307_2.png)

뭔 소린지 모르겠음





### 추가조사

- 제안: 

  - Geoffrey Hinton의 강의([강의록](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))에서 처음 제안됨. 

- 수식

  ![ch0307_13]({{site.url}}/images/$(filename)/ch0307_13.png)

- Root Mean Square Propagation 이름의 유래

  - Root Mean Square가 학습률 조정 계산에 사용됨. 여기서 유래된 이름. ($$ s_t $$가 그라디언트 제곱의 '이동평균')

  - 특정 파라미터의 그래디언트가 크게 발생하면, 해당 파라미터의 학습률은 감소 ($$ w_{t+1} $$ 수식에서, 그라디언트의 분모항으로 들어감. 제곱 후 루트 취해주므로 항상 양수)

  - 각 파라미터에 대해, 그래디언트의 제곱의 이동 평균을 유지한다.

    

- 핵심 아이디어

  - 각 파라미터에 대해 학습률을 개별적으로 조정
  - 그래디언트가 작은 파라미터는 더 큰 학습률을 가지며, 그래디언트가 큰 파라미터는 더 작은 학습률을 가짐

  

  

- 효과성

  - 이러한 동적 학습률 조정은 비등방성(anisotropic) 비용 함수에서의 최적화에서 성능 향상.

- 예시

  - 사람이 경사가 가파를 때는 작은 걸음을, 경사가 완만할 때는 큰 걸음을 걷는 것과 유사함.

- 논문x. 수학적 증명x 



>  w1, w2, ... 들에 대한 이동량을 normalize해주는 것 같다?



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

#### 등방한 경우

$$ Loss = w_1^2 + w_2^2 $$

##### 등방한 경우, SGD 지그재그로 움직이지 않음, 모멘텀 alpha 0.9, RMSProp beta 0.1

![ch0307_12]({{site.url}}/images/$(filename)/ch0307_12.png)

*연한 색: 초기 위치, 진한 색: 나중 위치, 빨간 색: 최종 위치*



##### 등방한 경우, SGD 지그재그로 움직이지 않음, 모멘텀 alpha 0.1, RMSProp beta 0.9

![ch0307_11]({{site.url}}/images/$(filename)/ch0307_11.png)





<br>

<br>

#### 비등방한 경우

#### 비등방 + SGD가 지그재그로 움직이는 경우 + 모멘텀 alpha 0.1

![ch0307_7]({{site.url}}/images/$(filename)/ch0307_7.png)





##### 비등방 + SGD가 지그재그로 움직이는 경우 + 모멘텀 alpha 0.9

$$ Loss = w_1^2 + 8*w_2^2 $$

![ch0307_6]({{site.url}}/images/$(filename)/ch0307_6.png)





##### 비등방 + SGD가 지그재그로 움직이는 경우 + 모멘텀 alpha 0.99 

수렴을 못하기도 하네. overshooting 발생

![ch0307_8]({{site.url}}/images/$(filename)/ch0307_8.png)



##### 비등방 + SGD가 지그재그로 움직이는 경우 + RMSProp beta 0.1 

![ch0307_9]({{site.url}}/images/$(filename)/ch0307_9.png)



##### 비등방 + SGD가 지그재그로 움직이는 경우 + RMSProp beta 0.99

![ch0307_10]({{site.url}}/images/$(filename)/ch0307_10.png)



모델과 설정 값에 따라, 수렴하는 양태가 다르다.

항상 지그재그가 효과적으로 없어지는 것도 아니다... 모델의 특성을 잘 보고 판단하여 optimizer도 튜닝해야 한다.



- 모멘텀을 사용할 때의 장점
  - 비등방 오류 표면에서, 지그재그 움직임이 줄어들고, 빠르게 수렴 가능 (튜닝은 잘 해야 함)
  - Local Minima 및 안장점 Saddle Point 탈출
  - 고차원 및 복잡한 모델에서도 잘 동작 함



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
        print(f'w1, w2 = ', w1, w2)
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
    return w1**2 + 8*w2**2

def gradient_w1(w1):
    return 2 * w1

def gradient_w2(w2):
    return 2 * 8 * w2

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
    min_alpha = 0.2  # 최소 alpha 값 설정
    colors = [mcolors.to_rgba(color, alpha=min(1, i/num_points + min_alpha)) for i in range(num_points)]
    ax.scatter(x, y, [f(w1, w2) for w1, w2 in path], c=colors, marker='o', s=30)
    ax.scatter(x[-1], y[-1], [f(w1, w2) for w1, w2 in [path[-1]]], c='red', marker='o', s=50)  # 최종 도착 지점을 다른 색으로 표시
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Loss Function')

ax1 = fig.add_subplot(131, projection='3d')
plot_path(ax1, sgd_path, 'SGD for $f(w1, w2) = w1^2 + 8*w2^2$', 'blue')

ax2 = fig.add_subplot(132, projection='3d')
plot_path(ax2, momentum_path, 'Momentum for $f(w1, w2) = w1^2 + 8*w2^2$', 'blue')

ax3 = fig.add_subplot(133, projection='3d')
plot_path(ax3, rmsprop_path, 'RMSProp for $f(w1, w2) = w1^2 + 8*w2^2$', 'blue')

plt.show()
```

