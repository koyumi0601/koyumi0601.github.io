---
layout: single
title: "AI Deep Dive, Chapter 5. 이진 분류와 다중 분류 01. 선형분류와 퍼셉트론"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 5 - 01. 선형분류와 퍼셉트론



##### 사전 추가 조사 내용

##### 선형 분류란?

선형 분류(Linear Classification)는 데이터 포인트를 두 개 이상의 클래스로 분류하는 방법 중 하나로, 데이터 포인트들을 선형 경계로 구분하는 방식을 의미합니다. 선형 분류는 데이터 포인트들 사이에 직선(2D), 평면(3D), 또는 초평면(higher dimensions)을 사용하여 클래스를 구분합니다.

선형 분류의 주요 특징은 다음과 같습니다:

1. **선형 경계**: 선형 분류는 데이터를 구분하는 경계가 선형이라는 특징이 있습니다. 이는 2D 공간에서는 직선, 3D 공간에서는 평면, 그 이상의 차원에서는 초평면으로 표현됩니다.
2. **간단하고 빠름**: 선형 분류는 계산적으로 간단하고 빠르게 수행될 수 있습니다. 따라서 대량의 데이터에 대해서도 효율적으로 처리할 수 있습니다.
3. **제한적인 표현력**: 선형 분류는 데이터가 선형적으로 구분 가능할 때 잘 작동합니다. 그러나 데이터가 복잡한 패턴을 가지고 있거나 선형적으로 구분되지 않을 때는 성능이 떨어질 수 있습니다.
4. **확장성**: 선형 분류는 커널 트릭과 같은 기법을 사용하여 비선형 데이터에 대해서도 적용할 수 있습니다.

대표적인 선형 분류 알고리즘에는 로지스틱 회귀(Logistic Regression), 서포트 벡터 머신(SVM)의 선형 버전, 퍼셉트론(Perceptron) 등이 있습니다.

##### 퍼셉트론이란?

퍼셉트론(Perceptron)은 인공 신경망의 가장 기본적인 형태로, 선형 분류기의 일종입니다. 1957년에 프랑크 로젠블라트(Frank Rosenblatt)에 의해 처음 소개되었습니다. 퍼셉트론은 입력 벡터에 가중치를 곱하고, 그 결과를 합산한 후 활성화 함수를 통과시켜 출력을 생성하는 구조를 가집니다.

퍼셉트론의 주요 특징 및 구조는 다음과 같습니다:

1. **입력과 가중치**: 퍼셉트론은 여러 개의 입력 $$ x_1, x_2,...,x_n$$ 을 받아들이며, 각 입력에는 가중치 $$ w_1, w_2,...,w_n$$ 이 연결되어 있습니다.
2. **가중합**: 입력과 가중치의 곱을 합산하여 가중합 $$ \sum{w_i x_i} $$을 계산합니다.
3. **활성화 함수**: 가중합은 활성화 함수(대표적으로 계단 함수)를 통과하여 출력 *y*를 생성합니다. 계단 함수는 임계값을 기준으로 입력이 임계값보다 크면 1, 작으면 0을 출력합니다.
4. **학습**: 퍼셉트론은 학습 데이터를 사용하여 가중치를 조정합니다. 잘못된 예측을 했을 경우, 가중치를 조정하여 오차를 최소화하는 방향으로 학습합니다.
5. **선형 분류**: 퍼셉트론은 선형적으로 구분 가능한 데이터에 대해서만 완벽하게 학습할 수 있습니다. XOR 문제와 같이 선형적으로 구분되지 않는 데이터에 대해서는 퍼셉트론이 수렴하지 않습니다.

퍼셉트론은 초기 인공 신경망 연구의 기초를 제공했으며, 현대의 딥 러닝과 다층 퍼셉트론(MLP)의 발전에 기여했습니다. 하지만 단순한 퍼셉트론은 한계가 있어, 실제 문제를 해결하기 위해서는 여러 개의 퍼셉트론을 연결한 다층 퍼셉트론(MLP) 구조를 사용합니다.





## 강의 본문

- nonlinear activation(unit step function)으로 이진분류 문제를 풀어보자
- ex. 입력: 키와 몸무게, 출력: 1(빼야할 사람) 또는 0(쪄야 할 사람)

![ch0501_2]({{site.url}}/images/$(filename)/ch0501_2.png)

![ch0501_1]({{site.url}}/images/$(filename)/ch0501_1.png)

```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.linspace(0, 200, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
Z = np.where(Y > X + 1, 1, 0)

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Need to Lose Weight')
ax.set_title('3D Plot of Weight vs Height')

# 각 축의 범위 설정
ax.set_xlim([0, 200])
ax.set_ylim([0, 100])
ax.set_zlim([-1, 2])

plt.show()
```



- 주의사항: 위의 예제처럼, 선형분류라고 용어를 쓰지만, 입력과 출력과의 관계는 선형적인 것은 아니다.
- 선형분류=경계가 선형



#### 강의 노트

- 선형 분류(경계가 선형)에 쓰일 수 있다
- hidden layer 없이 unit step function을 activation으로 사용하면 퍼셉트론이라 함
- 3D로 표현되는 것 확인 (입력과 출력과의 관계는 비선형)
- 단점
  - 미분불가 (gradient descent 못 씀)
  - 너무 빡빡하게 분류한다

- 극복방안
  - unit step function -> 대신 sigmoid 
    - $$ \frac{1}{1+e^{-x}} $$ (수식)
    - 0에서 0.5를 지난다. 양수는 1로 수렴, 음수는 0으로 수렴
    - 최대 기울기 1/4 at (0, 0.5)
    - 전구간 미분가능
    - 좀 더 부드러운 분류 가능함

![ch0501_3]({{site.url}}/images/$(filename)/ch0501_3.png)



![ch0501_4]({{site.url}}/images/$(filename)/ch0501_4.png)

이 출력을, 확률 또는 정도라는 말로 해석할 수 있도록 해 줌.

출력이 0.3이면, 한 30% 정도로 뺄 필요가 있을 가능성이 있는 사람이라는 뜻으로 얘기해줄 수 있다

가장 멀리 찢어 놓는 합리적인 분류 기준 선을 찾게 됨

- 어떤 선이 더 좋은 분류선일까?

  ![ch0501_5]({{site.url}}/images/$(filename)/ch0501_5.png)

- 퍼셉트론은, 1번이나 2번이나 둘 다 백점임. 확실히 나눠주므로.

- 반면, sigmoid를 사용했다면, 1번 선은 데이터 3이 경계선에 놓이므로, 애매함, 2번 선으로 가기 위해 업데이트를 할 것. -> 가장 멀리 찢어놓는 선을 찾는다라는 것.
