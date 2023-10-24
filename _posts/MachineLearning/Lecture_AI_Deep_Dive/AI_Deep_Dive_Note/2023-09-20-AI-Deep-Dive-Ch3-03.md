---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 03. Gradient descent 경사하강법"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*




# Chapter 3 - 03. Gradient descent 경사하강법

- 일단, 처음 a, b는 아무렇게나 정한다
- 현재 a, b 위치에서, L을 줄이는 방향으로 나아가자

![ch0303_1]({{site.url}}/images/$(filename)/ch0303_1.png)

- 방향은 어디로 가지? 마침 Gradient는 항상 가장 가파른 방향을 향한다. 
- 최소인 곳으로 갈 거니까, 반대 방향으로 가면 됨. (빼기)

![ch0303_2]({{site.url}}/images/$(filename)/ch0303_2.png)

- Learning rate $$ \alpha $$ (보폭)의 존재 이유?

  - 상수 혹은 스케줄링

  - 예시

    ![ch0303_3]({{site.url}}/images/$(filename)/ch0303_3.png)

    - 방향은 맞는데, 생각보다 Gradient가 커서, 생각보다 많이 가서, 수렴하지 않는 경우가 발생할 수 있다.

  - $$ 0 <  \alpha < 1 $$ .

    ![ch0303_4]({{site.url}}/images/$(filename)/ch0303_4.png)

    ![ch0303_5]({{site.url}}/images/$(filename)/ch0303_5.png)

- Initial weight도 잘 잡아야 함. (LeCun, Xavier, He)

- Gradient descent의 단점

  - 너무 신중하게 방향 선택
    - 모든 데이터의 loss 값을 다 고려해서 미분 -> 한번 구하는데 한시간 걸림
    - local minimum에 빠질 수 있다.

![ch0303_6]({{site.url}}/images/$(filename)/ch0303_6.png)



- Result
  - a와 b에 따른 loss함수를 $$ x^2 + y^2 $$ 이라고 설정하여, 최소값을 향해 나아가면 아래와 같은 궤적이 그려진다. 여기서 learning rate는 0.1로 설정했다.

![ch0303_7]({{site.url}}/images/$(filename)/ch0303_7.png)





>  **경사 하강법(Gradient Descent)**은 최적화 문제에서 최솟값을 찾는 데 사용되는 반복적인 최적화 알고리즘입니다. 주로 머신 러닝 및 딥 러닝에서 모델을 훈련할 때 매개 변수를 조정하는 데 사용됩니다. 이 알고리즘은 다음과 같이 동작합니다:
>
> <br>
>
> 1. **초기화**: 먼저 매개 변수를 초기화합니다. 일반적으로 무작위로 초기화하거나 특정한 값을 설정합니다. <br>
> 2. **손실 함수 계산**: 최적화하려는 함수(손실 함수 또는 비용 함수)를 계산합니다. 이 함수는 주어진 매개 변수에서 모델의 성능을 나타냅니다. 목표는 이 손실 함수를 최소화하는 매개 변수를 찾는 것입니다. <br>
> 3. **기울기 계산**: 현재 매개 변수 위치에서 손실 함수의 기울기(경사)를 계산합니다. 이 기울기는 현재 위치에서 손실 함수가 가장 가파르게 증가하는 방향을 나타냅니다. <br>
> 4. **매개 변수 업데이트**: 기울기의 반대 방향으로 매개 변수를 업데이트합니다. 이 방향으로 이동하면 손실 함수 값을 줄일 수 있습니다. 업데이트할 때 사용되는 속도(학습률)는 하이퍼파라미터로 조절할 수 있으며, 이 값이 너무 작으면 최소값에 수렴하기까지 시간이 오래 걸릴 수 있고, 너무 크면 발산할 수 있습니다. <br>
> 5. **반복**: 위 단계를 반복합니다. 일반적으로 일정 횟수(epoch)나 일정한 손실 값으로 수렴할 때까지 반복합니다. <br>
> 6. **수렴**: 알고리즘이 수렴하면(기울기가 거의 0인 지점에 도달하면) 최적의 매개 변수 값을 찾게 됩니다. <br>
>
> 경사 하강법은 다양한 변형이 존재하며, 주로 배치 경사 하강법(Batch Gradient Descent), 확률적 경사 하강법(Stochastic Gradient Descent), 미니 배치 경사 하강법(Mini-Batch Gradient Descent) 등이 있습니다. 각각의 변형은 데이터의 양과 계산 리소스에 따라 선택됩니다.