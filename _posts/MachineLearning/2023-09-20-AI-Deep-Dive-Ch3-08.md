---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 08. Adam (Adaptive Moment Estimation)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 08. Adam (Adaptive Moment Estimation)

- 앞선 Momentum, RMSProp를 합친 것

- 논문 [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf%5D)

- 아래의 수식을 보면 알 수 있다.

![ch0308_5]({{site.url}}/images/$(filename)/ch0308_5.png)

논문의 제일 아래 부분에 $$ \theta_t $$를 관찰해보면, 그동안 구해봤던(빨간색) 형태의 업데이트 값에 그라디언트와 learning rate 계산 형태를 그대로 따름을 알 수 있다.

![ch0308_6]({{site.url}}/images/$(filename)/ch0308_6.png)

이 적용 값에 모멘텀과 RMS Prop를 반영을 어떻게 했는 지 알아보면 된다.

- $$ m_t $$:

  - '방향'을 exponential moving average를 통해서 구하는 부분이다.

  - Exponential Moving Average, EMA: 

  - 시계열 데이터의 평균을 계산할 때, 최근의 데이터에 더 많은 가중치를 부여하는 방법.

  - $$ EMA_t = β ⋅ EMA_{t−1} + (1−β) ⋅ x_t $$ .
  - ![ch0308_7]({{site.url}}/images/$(filename)/ch0308_7.png)
  - 그래디언트를 누적함으로써 관성을 얻게 하는 것.

- $$ v_t $$:
  - 보폭
  - RMSProp을 구현한 것
  - 제곱을 누적하고, 최종결과물에서 sqrt로 나누어줌. (항상 양수-크기로 나눠주려고)
  - (표기 관점에서, 제곱으로 쓴 것은, 편의상 각 element의 제곱으로 하자-논문)
  - ![ch0308_8]({{site.url}}/images/$(filename)/ch0308_8.png)

- $$ \hat{m_t} $$:
  - 모멘텀의 개념을 넣기 위해서
  - 목적: $$ E[\hat{m_t}] $$과 $$ E[g_t]$$(기댓값)이 유사해지도록 조절해 주는 것
  - ![ch0308_9]({{site.url}}/images/$(filename)/ch0308_9.png)
  - 그냥 $$ m_t $$에 대해서 평균을 취하면, $$ g_t $$랑 비슷하지 않음
  - 그래서 $$ \hat{m_t} $$으로 바꿔주고 평균을 취해서 $$ g_t $$랑 비슷한 기댓값을 가지도록 조절해줬다.
  - 맨 처음에는 $$ m_0 = v_0 = 0 $$에서 시작 (초기 위치에서는 이전 값을 모르니까 0). 어쩔 수 없음. 근데 이 부분을 보정해주는 것.
  - ![ch0308_11]({{site.url}}/images/$(filename)/ch0308_11.png)

- $$ \hat{v_t} $$:

  - RMSProp의 개념을 넣기 위해서

  - ![ch0308_10]({{site.url}}/images/$(filename)/ch0308_10.png)

    

- $$ \epsilon $$:

  - 작은 양수
  - 나누기 0이 되지 않도록 방지. 값이 확 튐
  - default $$ 10^{-12} $$



### 정리하자면,

![ch0308_12]({{site.url}}/images/$(filename)/ch0308_12.png)







![ch0308_3]({{site.url}}/images/$(filename)/ch0308_3.png)



<br> <br>



# 추가 조사

##### optimizers

- SGD:
  - 안장점에서 취약
- Momentum
- NAG (Nesterov Accelerated Gradient):
  - Momentum 방법의 변형
  - 기본 아이디어
    - 현재 그래디언트가 아닌, 모멘텀이 적용된 예상 위치에서의 그래디언트를 사용하여 파라미터를 업데이트
    - 일반적인 모멘텀 방법보다 더 빠르게 수렴
- Adgrad (Adaptive Gradient Algorithm):
  - 각 파라미터에 대해 학습률을 개별적으로 조정
  - 자주 발생하는 특성에 대해서는 학습률을 낮추고, 드물게 발생하는 특성에 대해서는 학습률을 높입
  - 희소한 데이터에 효과적
  - 그러나, 학습률이 너무 빨리 감소하여 학습이 일찍 멈출 수
- Adadelta
  - Adagrad의 단점을 해결하기 위한 알고리즘
  - Adadelta는 고정된 학습률을 사용하지 않고, 과거의 모든 그래디언트의 제곱의 평균을 사용하여 학습률을 동적으로 조정
  - Adadelta는 Adagrad보다 더 안정적인 학습을 제공
- RmsProp



