---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 02. 선형 회귀, 개념부터 알고리즘까지 step by step"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*




# Chapter 3 - 02. 선형 회귀, 개념부터 알고리즘까지 step by step



ax+b를 인공신경망 나타낸 것은 아래와 같다



![ch0302_1]({{site.url}}/images/$(filename)/ch0302_1.png)



이를 이용해서 선형회귀를 해볼 수 있다.







### 선형 회귀란?

- 입력과 출력 간의 관계를 *선형*으로 놓고 추정하는 것 

- 분류: 지도학습

- 목적: 처음보는 입력에 대해서도 적절한 출력을 얻기 위함

- 예시:
  - 키와 몸무게의 관계를 ax+b로 놓고 a, b를 잘 추정해서, 처음보는 키에 대해서도 적절한 몸무게를 출력하는 머신을 만들어보자


![ch0302_2]({{site.url}}/images/$(filename)/ch0302_2.png)

- 알아내야 할 것: *최적*의 웨이트, 바이어스
- 주어진 것: 데이터 기반
- Criteria: Loss 함수(혹은 cost)
  - 최적의 a,b? 내가 고른 a, b가 좋다 나쁘다를 판단할 수 있어야 함. 이를 판단하기 위한 근거, criteria = loss 함수
  - loss를 최소화할 수 있는 weight, bias를 찾아내야 한다.
  - loss함수를 잘 정의내려야 한다.
  - 머신의 예측 $$ \hat{y} $$과 실제 몸무게 y의 차이로 loss를 정의해보자



![ch0302_4]({{site.url}}/images/$(filename)/ch0302_4.png)

- 데이터 y = y1, y2, y3, y4, y5
- 실제값(y#)과 머신러닝의 모델값($$\hat{y}$$)의 차이 e를 loss라고 하자.

- e5 뿐만 아니라, 다른 데이터에 대해서도 모두 합해보자
  - 그냥 더하면 안된다. 양수로 더해야한다.
  - 절대값 vs 제곱값
    - 차이를 인지하고 사용해야 한다.
    - 에러가 점점 커질 수록, 제곱이 더 민감하다
    - 절대값은 2일 때 미분을 따로 예외처리 해줘야 한다. (=0). 제곱은 미분 가능하다.
    - MSE(Mean Squared Error), 평균 안내줘도 최소값은 찾을 수 있으므로, 생략가능하다.

![ch0302_5]({{site.url}}/images/$(filename)/ch0302_5.png)

- L을 최소화하는 a, b 어떻게 찾지?
  - a, b를 일일이 바꿔가며 L값을 그래프로 그려보자



```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.array([[160, 45], [162, 51], [165, 55], [170, 63], [180, 75]], dtype=np.float32)
x_train = data[:, 0]  # 키
y_train = data[:, 1]  # 몸무게

# 주어진 범위 내에서 a와 b의 조합을 시도하며 MSE 값을 저장
a_range = np.arange(0, 10, 0.1)
b_range = np.arange(-1000, 100, 1)
mse_values = np.zeros((len(a_range), len(b_range)))

for i, a in enumerate(a_range):
    for j, b in enumerate(b_range):
        # 현재 a와 b를 사용하여 예측값 계산
        y_pred = a * x_train + b

        # MSE 계산
        mse = np.mean((y_pred - y_train) ** 2)
        mse_values[i, j] = mse

# a와 b 값에 대한 그리드 생성
a_grid, b_grid = np.meshgrid(a_range, b_range)

# 최소 MSE 값을 찾아 해당 위치 저장
min_mse = np.min(mse_values)
min_mse_idx = np.unravel_index(np.argmin(mse_values, axis=None), mse_values.shape)
min_a = a_range[min_mse_idx[0]]
min_b = b_range[min_mse_idx[1]]

# 3D surface plot으로 MSE 값 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(a_grid, b_grid, mse_values.T, cmap='viridis')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('MSE Surface Plot')

# 최소 MSE 지점을 점으로 표시
ax.scatter(min_a, min_b, min_mse, color='red', marker='o', s=100, label='Min MSE')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()
plt.show()

# 최소 MSE를 갖는 a와 b 값 출력
print(f"Min MSE: {min_mse}")
print(f"Min a: {min_a}")
print(f"Min b: {min_b}")
```

- MSE

![ch0302_7]({{site.url}}/images/$(filename)/ch0302_7.png)



- Model
  - Min MSE: 2.049999952316284, 0보다 항상 크다. 제곱의 합이라
  - Min a: 1.5
  - Min b: -193

![ch0302_6]({{site.url}}/images/$(filename)/ch0302_6.png)



- 지금은 모델이 작으니까 썼지, 노드, 변수가 몇 억개면? 반복문이 너무 많아진다.  좀 더 스마트한 방법이 필요하다 -> Gradient descent!