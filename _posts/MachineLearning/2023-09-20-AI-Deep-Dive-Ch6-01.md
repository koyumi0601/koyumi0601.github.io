---
layout: single
title: "AI Deep Dive, Chapter 6. 인공신경망 그 한계는 어디까지인가? 01. Universal Approximization Theorem (왜 하필 인공신경망인가)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 6 - 01. Universal Approximization Theorem (왜 하필 인공신경망인가)

- Universal Approximization Theorem
  - 한계가 없다는 것을 이론적으로 보여줌
  - MLP: 행렬 곱하고 벡터 더하고 activation, ... 이런 함수다
  - $$ f(f(xW_1 + b_1)W_2+b_2)) $$. (1차 함수에 대하여 activation)
  - 굳이 이런 형태로 함수를 표현해야 하는 이유? x^Tx = x^2 이런 건 왜 안쓰나? 히든 레이어 딱 한층만 있어도 어떤 연속 함수든 나타낼 수 있기 때문. 즉, loss를 딱 0으로 만들어 버릴 수 있다.



## 시각적 증명 웹사이트

- A visual proof that newral nets can compute any function [http://neuralnetworksanddeeplearning.com/chap4.html](http://neuralnetworksanddeeplearning.com/chap4.html)









## 추가 조사

- Universal Approximation Theorem

  - Universal Approximation Theorem (UAT)은 인공 신경망의 강력한 특성 중 하나를 설명하는 중요한 이론이다. 이 이론은 특정 조건 하에서 단일 은닉층을 가진 인공 신경망이 임의의 연속 함수를 근사할 수 있음을 의미한다. 주어진 충분한 수의 뉴런과 적절한 활성화 함수를 사용하면, 신경망은 복잡한 함수도 모델링할 수 있다.

    UAT의 주요 내용은 다음과 같다:

    1. **한 개의 은닉층**: 신경망이 단 하나의 은닉층만 가지고 있어도, 주어진 충분한 수의 뉴런과 적절한 활성화 함수를 사용하면 임의의 연속 함수를 근사할 수 있다.
    2. **활성화 함수**: 이론은 특정 활성화 함수, 예를 들어 시그모이드 함수나 ReLU와 같은 비선형 활성화 함수를 사용할 때 성립한다. 선형 활성화 함수만 사용하는 경우에는 이 이론이 성립하지 않는다.
    3. **근사의 정밀도**: 신경망의 뉴런 수가 늘어날수록 근사의 정밀도가 향상된다. 더 많은 뉴런을 사용하면 더 정확한 근사를 얻을 수 있다.
    4. **실제 구현**: 이론적으로는 한 개의 은닉층만으로도 임의의 함수를 근사할 수 있지만, 실제로는 깊은 신경망 (여러 은닉층을 가진 신경망)이 훨씬 더 효과적인 경우가 많다. 이는 깊은 신경망이 동일한 근사 정도를 달성하기 위해 필요한 뉴런 수가 훨씬 적기 때문이다.

    UAT는 인공 신경망의 능력을 이론적으로 증명하는 것이지만, 실제 문제에서 최적의 신경망 구조나 학습 알고리즘을 찾는 것은 별개의 문제다. UAT는 신경망이 임의의 함수를 근사할 수 있는 능력을 보여주지만, 실제로 그러한 함수를 학습하는 것은 다양한 요인에 따라 달라질 수 있다.



### 수학적 증명, 개요

**Theorem**: 활성화 함수 *σ* (예: 시그모이드 함수)를 가진 단일 은닉층을 가진 피드포워드 신경망은 임의의 연속 함수 *f*를 근사할 수 있다. 즉, 모든 *x*에 대해, 충분한 은닉층 뉴런 수 N와 적절한 가중치와 편향을 가진 신경망 *F*(*x*)가 존재하여 ∣*F*(*x*)−*f*(*x*)∣<*ϵ* (여기서 ϵ은 아주 작은 양의 값)이 성립한다.

**증명의 개요**:

1. **Basis Function**: 활성화 함수 *σ*를 사용하여 "기저 함수"를 구성할 수 있다. 이 기저 함수는 입력 공간의 임의의 부분을 "포착"할 수 있다. 이를 통해 신경망이 임의의 함수를 근사하는 데 필요한 "블록"을 형성할 수 있다.
2. **Combination of Basis Functions**: 이러한 기저 함수들을 조합하여 복잡한 함수를 형성할 수 있다. 가중치와 편향을 조절하여 이 기저 함수들의 합을 통해 원하는 함수를 근사할 수 있다.
3. **Arbitrarily Close Approximation**: 충분한 수의 은닉층 뉴런을 사용하면, 신경망의 출력은 원하는 함수에 아주 가깝게 근사될 수 있다. 이는 각 뉴런이 특정 부분을 근사하고, 이러한 부분들이 합쳐져 전체 함수를 근사하기 때문이다.



### 직관적인 증명

- https://hanlue.tistory.com/12

- UAT의 기본 아이디어는 간단한 함수들의 조합으로 복잡한 함수를 근사화하는 것입니다. 예를 들어, sigmoid 함수는 다음과 같이 정의됩니다:

  $$ \sigma(x) = \frac{1}{1+exp(-(wx-b))} $$

  수식에서 b 를 적절하게 셋팅하여 sigmoid function 의 중심 (y 값이 0이 되는 부분) 을 이동시키고 w 를 크게 셋팅하면 위와 같이 가파른 sigmoid function 을 만들 수 있다. 

  sigmoid 함수는 0과 1 사이의 값을 가집니다. 여러 sigmoid 함수를 조합하면 다양한 형태의 함수를 생성할 수 있습니다. 예를 들어, 두 개의 sigmoid 함수를 뺄셈 연산으로 조합하면 "step" 함수와 유사한 형태를 만들 수 있습니다.

  neural network 를 이용하여 sigmoid function 의 뺄셈 연산을 어떻게 만들어낼지?  w 와 b 가 적절히 셋팅되면 가파른 모양의 sigmoid function 이 나오는 것을 확인하였다.

  마지막 layer 에, 위와 같은 sigmoid function 값을 이용하여 뺄셈을 할 수 있는 weight를 설정해 준다면 사각형 모양의 simple function 을 만들 수 있다.

- 강의에서는, x축(노드의 입력값)을 scaling함으로써 unit step function처럼 사용하는 것을 가이드.



#### 용어

- Feed-forward neural networks

- 

- 튜링완전성(Turing completeness)

- 튜링기계(Turing machine)

  - 튜링 기계(Turing machine)는 1936년에 앨런 튜링(Alan Turing)에 의해 제안된 이론적인 계산 모델입니다. 튜링 기계는 모든 알고리즘을 시뮬레이션할 수 있는 능력을 가진다고 여겨지며, 따라서 현대 컴퓨터 과학의 기초를 이루는 중요한 개념입니다.

    튜링 기계의 주요 구성 요소는 다음과 같습니다:

    1. **무한한 길이의 테이프(Tape)**: 각각의 칸에는 하나의 기호가 쓰여질 수 있으며, 초기에는 모든 칸이 특정 기호(보통 '빈칸'을 의미하는 기호)로 채워져 있습니다.
    2. **테이프 헤드(Head)**: 테이프의 특정 위치를 가리키며, 현재 위치의 기호를 읽거나 쓸 수 있습니다.
    3. **상태 기계(State machine)**: 현재 상태를 나타내며, 테이프 헤드가 읽은 기호와 현재 상태에 따라 다음 상태를 결정하고, 테이프에 쓸 기호를 결정하며, 테이프 헤드의 이동 방향을 결정합니다.

    튜링 기계는 주어진 입력에 대해 계산을 수행하며, 계산이 완료되면 특정 상태(보통 '정지 상태'라고 함)에 도달합니다. 튜링 기계의 능력은 매우 강력하여, 튜링 기계로 표현 가능한 모든 알고리즘은 실제 컴퓨터로도 구현 가능하다고 여겨집니다.

    이러한 튜링 기계의 개념은 "튜링 완전성(Turing completeness)"이라는 중요한 개념의 기초가 되며, 어떤 시스템이 튜링 완전하다는 것은 그 시스템이 튜링 기계와 동일한 계산 능력을 가지고 있음을 의미합니다.





### 실험

