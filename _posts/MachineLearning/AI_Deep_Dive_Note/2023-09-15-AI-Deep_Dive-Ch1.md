---
layout: single
title: "AI Deep Dive, Chapter 1. 딥러닝을 위한 필수 기초 수학"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*AI Deep Dive Note*


# OT

- Chapter 1 - 딥러닝을 위한 필수 기초 수학
- Chapter 2 - 왜 현재 AI가 가장 핫할까?
- Chapter 3 - 왜 인공 신경망을 공부해야 하는가?
- Chapter 4 - 딥러닝, 그것이 알고 싶다.
- Chapter 5 - 이진 분류와 다중 분류
- Chapter 6 - 인공 신경망, 그 한계는 어디까지인가?
- Chapter 7 - 깊은 인공 신경망의 고질적 문제와 해결 방안
- Chapter 8 - 왜 CNN이 이미지 데이터에 많이 쓰일까?
- Chapter 9 - 왜 RNN보다 트랜스포머가 더 좋다는 걸까?

# 01-01 함수와 다변수 함수


한 개 입력, 한 개 출력

$$ x \rightarrow  f \rightarrow  f(x) = y = x^2 $$

여러 개 입력, 한 개 출력

$$ f(x, y) = z = yx^2 $$


한 개 입력, 여러 개 출력

$$ x \rightarrow [f1, f2] = [x, x^2] $$


두 개 입력, 두 개 출력

$$ z = f(x, y) = [xy^2, x+y] $$

# 01-02 로그 함수

- 지수함수의 역함수

$$ log_{10} 100 = 2 $$

$$ log_{e} e^3 = 3 $$

- 로그 함수의 성질

$$ log_a xy = log_a x + log_a y $$

$$ log_a x^n = n log_a x $$

$$ log_{a^m} x = \frac{1}{m} log_a x $$

$$ log_a b = \frac{ log_c b}{log_c a} $$

$$ log_a b = \frac{1}{log_b a} $$

$$ a^{log_a x} = x $$

$$ a^{log_b c } = c^{log_b a} $$

# 01-03 벡터와 행렬

- 열 벡터, 행 벡터, 행렬, 성분
- 연립 1차 방정식

$$ \left\{\begin{matrix}
x+2y = 4\\ 
2x+5y = 9
\end{matrix}\right. $$

$$ \begin{bmatrix}
1 & 2 \\ 
2 & 5
\end{bmatrix}
 \begin{bmatrix}
x \\ 
y
\end{bmatrix}
= 
\begin{bmatrix}
4 \\ 
9
\end{bmatrix}
$$
- 행렬과 벡터의 곱

- 벡터의 방향과 크기

$$ [2, 1] $$

시점이 달라도, 방향과 크기가 같으면 같은 벡터이다.

크기 
- 벡터의 길이
    - l2 norm $$ ({2^2 + 1^2})^{\frac{1}{2}} $$
    - l1 norm $$ ({2^1 + 1^1})^1 $$

방향
- 사잇각

# 01-04 전치와 내적

- Transpose

$$ A^T_{ij} = A_{ji} $$
$$

( \begin{bmatrix}
a_{1 1} & a_{1 2}  \\ 
a_{2 1} & a_{2 2}
\end{bmatrix}

 \begin{bmatrix}
x_1  \\ 
x_2
\end{bmatrix}
)^T
=
{
 \begin{bmatrix}
b_1  \\ 
b_2
\end{bmatrix}
}^T
$$

$$

[x_1 x_2]
 \begin{bmatrix}
a_{1 1} & a_{2 1}  \\ 
a_{1 2} & a_{2 2}
\end{bmatrix}

=
[b_1 b_2]

$$
- 내적

    - 내적은 닮은 정도를 나타낸다

    $$ \left \| a \right \| \left \| b \right \| cos \theta $$

    - 사잇각이 같은 경우, cos 0 최대
    - cos 90인 경우, 0
    - 반대 방향인 경우, 반대로 닮음.

# 01-05 극한과 입실론-델타 논법

- $$ \lim_{x \rightarrow a} f(x) $$ : x가 a와 무진장 가까운 값일 때, f(x)는 뭐랑 무진장 가깝냐?
- 다가간다든가, 움직이는 것 아님
- $$ \lim_{x \rightarrow a} f(x) = L $$을 만족하는 것을 그래프로 봅시다
- $$ \lim_{x \rightarrow a} f(x) = L: L $$ 주변 갭으로 어떤 양수 $$ \epsilon  $$ 을 잡더라도 요 갭 안으로 싹다 보내버릴 수 있는 a 주변 갭 $$ \delta $$가 존재하면 a에서의 극한 값은 L이다.

![epsilon_delta]({{site.url}}/images/$(filename)/epsilon_delta-1695015146024-2.png)



# 01-06 미분과 도함수

- 미분: 순간 기울기

![diff]({{site.url}}/images/$(filename)/diff.png)

극한을 이용해서 구할 수 있다.



$$ x^n \rightarrow n x^{n-1} $$



$$ e^x \rightarrow e^x $$



<center> $$ ln x \rightarrow \frac{1}{x} $$ & $$ log_2 x \rightarrow \frac{1}{ln_2}\frac{1}{x} $$ </center>



$$ f(x) + g(x) \rightarrow f'(x) + g'(x) $$



$$ af(x) \rightarrow af'(x) $$



$$ f(x)g(x) \rightarrow f'(x)g(x) + f(x)g'(x) $$



# 01-07 연쇄 법칙

![chain_rule]({{site.url}}/images/$(filename)/chain_rule.png)



# 01-08 편미분과 그라디언트

편미분: 다른 변수를 상수 취급

![partial_diff]({{site.url}}/images/$(filename)/partial_diff.png)

그라디언트: 벡터로 묶은 것

$$   \begin{bmatrix}
\frac{\delta f}{ \delta x}  \\
\frac{\delta f}{ \delta y}
\end{bmatrix}
$$



한 점에서의 그라디언트 구하기: 편미분하고 값을 대입





# 01-09 테일러 급수

![tayler]({{site.url}}/images/$(filename)/tayler.png)

![tayler2]({{site.url}}/images/$(filename)/tayler2.png)

C0, C1, C2 구하기

- 양변을 미분한 후 x에 0 대입

일반화하면

![tayler3]({{site.url}}/images/$(filename)/tayler3.png)



$$ e^x, ln x $$



![tayler4]({{site.url}}/images/$(filename)/tayler4.png)



# 01-10 스칼라를 벡터로 미분하는 법

![scalar_vector_diff]({{site.url}}/images/$(filename)/scalar_vector_diff.png)

![scalar_vector_diff_2png]({{site.url}} /images/$(filename)/scalar_vector_diff_2.png)

![scalar_vector_diff_3]({{site.url}}/images/$(filename)/scalar_vector_diff_3.png)

![scalar_vector_diff_4]({{site.url}}/images/$(filename)/scalar_vector_diff_4.png)





# 01-11 왜 그라디언트는 가장 가파른 방향을 향할까







learning rate가 필요한 이유: 근사를 활용하기 때문에 너무 많이 움직이면 근사가 깨짐



![gradient_stiff_2]({{site.url}}/images/$(filename)/gradient_stiff_2.png)



테일러 급수를 이용한 증명

![gradient_stiff_3]({{site.url}}/images/$(filename)/gradient_stiff_3.png)



강의 내용 필기



![Note2]({{site.url}}/images/$(filename)/Note2.png)





# 01-12 벡터를 벡터로 미분하는 법



- back propagation을 간단히 표현하는 데에 도움

![note4]({{site.url}}/images/$(filename)/note4.png)

![note5]({{site.url}}/images/$(filename)/note5.png)

![note6]({{site.url}}/images/$(filename)/note6.png)



# 01-13 벡터를 벡터로 미분할 때의 연쇄 법칙

![note7]({{site.url}}/images/$(filename)/note7.png)



# 01-14 스칼라를 행렬로 미분하는 법

![note8]({{site.url}}/images/$(filename)/note8.png)

![note9]({{site.url}}/images/$(filename)/note9.png)

# 01-15 행렬을 행렬로, 벡터를 행렬로 미분하는 법

행렬을 벡터로 변환한 후, 벡터 미분을 활용한다.

![note10]({{site.url}}/images/$(filename)/note10.png)



# 01-16 랜덤 변수와 확률 분포



랜덤 변수는 함수다

입력: 사건 (동전의 앞면 1, 뒷면 0)

출력: 실수의 값 

함수: 실수 값을 확률 값으로 바꿔주는 것이 확률 함수, 앞면 1 -> 1/2, 뒷면 0 -> 1/2

확률 함수의 종류

	- 확률 질량 함수: 동전의 앞면, 뒷면 처럼 딱딱 떨어질 때
	- 확률 밀도 함수: 대한민국 남성의 키 평균, 동전처럼 딱딱 떨어지지 않는 것

![pmf_pdf]({{site.url}}/images/$(filename)/pmf_pdf.png)



![note11]({{site.url}}/images/$(filename)/note11.png)





# 01-17 평균과 분산

확률 분포를 설명하는 두 가지 대푯값

### 평균

- mean: 수학적 언어
  - 산술평균, 기하평균, 조화평균

- average: 일상적 언어

- expectation: 기댓값

  - 주사위를 다섯번 던져서 나온 값들 

    $$ E[X] = \sum_i x_i p_i $$

- 연속 랜덤 변수

  $$ E[X] = \int_{+\inf}^{-\inf} xp(x)dx, E[X] = \mu $$

  - x를 곱한 상태로 적분한 것

  - 동전의 앞면 1, 확률 1/2, 뒷면 0, 확률 1/2, 모두 더하면 기댓값은 1/2

  - 보통은 중심 값이 나온다







### 분산

- 편차의 제곱의 평균
- 편차의 절대값은 분포에 대한 정확도가 떨어진다. 범위가 다름.



- 예시
  - 시험 점수 A = [100, 0], B = [50, 50]
  - 평균에 대한 차이, 편차를, 양수로 만들어준 후 모두 더해 줌, 평균 냄
  - 평균 50
- 이산 확률 일 때, $$ V[X] = \sum_i (x_i - \mu)^2 p_i $$
- 연속 확률 일 때, $$ = \int (x-\mu)^2 p(x) dx $$

- 다르게 쓰자면, 편차의 제곱의 평균 

  $$ E[(X-\mu)^2] $$

  $$ E[x^2 - 2x\mu + \mu^2] $$

  $$ E[x^2] - 2\mu E[x] + \mu^2 $$

  $$ E[x] = \mu $$ 이므로

  $$ E[x^2] - \mu^2 $$

  - 곧, 분산은 제곱 기댓값 빼기 평균 제곱으로 구할 수 있다



### 표준편차

- sqrt(V[X])
- 단위를 맞춰주기 위함, 분산은 단위가 면적과 혼동을 줌 $$ cm^2 $$



# 01-18 균등 분포와 정규 분포

딥러닝에서 아주 많이 씀



### 균등 분포

- uniform distribution

- 주사위, 동전 던지기, 연속적으로도 마찬가지
- X ~ U 라고 표기 함. X가 Uniform distribution을 따른다.
- 확률은 $$ \frac{1}{b-a} $$

![uniform_dist]({{site.url}}/images/$(filename)/uniform_dist.png)



### 정규 분포

![normal_dist]({{site.url}}/images/$(filename)/normal_dist.png)





# 01-19 최대 우도 추정 (MLE)

- 인공신경망은 MLE 기계이다

- Maximum Likelihood Estimation
  - 조건부 확률 vs likelihood 비교



- 우도 함수 Likelihood Function: 

  - $$ L(\theta; x) $$는 매개변수 $$\theta $$가 주어졌을 때 데이터 x가 나올 확률

  $$ L(\theta; x) = P(X = x; \theta) $$

- 로그 우도 (Log-Likelihood)

  - 우도 함수에 로그를 취한 것. 로그는 단조 증가 함수이므로, 로그 우도를 최대화하는 것은 원래의 우도를 최대화하는 것과 같다.

    $$ l(\theta; x) = log L(\theta; x) $$

- 최대 우도 추정치 (Maximum Likelihood Estimation)

  - 우도 함수 또는 로그 우도 함수를 최대화하는 매개변수 $$ \hat{\theta} $$ 

    $$ \hat{\theta} = arg_{\theta} max L(\theta; x) $$

    $$ \hat{\theta} = arg_{\theta} max l(\theta; x) $$

- 이산 확률 분포에서의 최대 우도 추정해 보기

![mle_1]({{site.url}}/images/$(filename)/mle_1.png)

![MLE]({{site.url}}/images/$(filename)/MLE.png)

*MLE를 한다 = likelihood를 보고(확률 값을 비교하여) 가장 크게 끔하는 조건을 고르는 것*



우도가 최대가 되는 x값을 찾으면, 그 x값이 가장 그럴듯한 매개변수나 상태라고 판단하는 것입니다. 이는 확률론적 관점에서의 추정이며, 이 x값이 바로 우리가 찾고자 하는 "최적의 추정치"입니다.



- 연속 확률 분포에서의 최대 우도 추정해보기

 ![MLE3]({{site.url}}/images/$(filename)/MLE3.png)

![MLE4]({{site.url}}/images/$(filename)/MLE4.png)

![mle5]({{site.url}}/images/$(filename)/mle5.png)



- 동전 던지기를 이용한 최대 우도 추정

![coin]({{site.url}}/images/$(filename)/coin.png)

![coin2]({{site.url}}/images/$(filename)/coin2.png)

측정된 데이터를 통해서, likelihood를 최대화하는 확률을 추정했다





# 01-20 최대 사후 확률 (MAP)

MAP(Maximum A Posteriori)는 사후 확률(posterior probability)을 최대화하는 방법으로, 우도(likelihood)와 사전 확률(prior distribution)을 모두 고려합니다.



1. **MLE (최대우도추정)**: "만약 측정값이 z이고 조건이 x라면, 우도(likelihood)는 측정값이 이렇게 나올 확률입니다. 이 경우, 우리는 x의 값을 바꿔가면서(입력 변수 x) 그에 따른 확률 밀도값(출력 함수)을 살펴봅니다. 그리고 그 중에서 가장 큰 확률 밀도값을 주는 x를 선택합니다."
   - $$ \hat{x} = arg_x max P(z|x) $$
2. **MAP (최대 사후 확률 추정)**: "측정값이 z로 주어져 있을 때, 우리는 x에 대한 확률 밀도값을 고려합니다. 이 확률 밀도값은 사전 확률과 측정값을 결합한 것입니다. 그래서, 이 사후 확률을 최대화하는 x를 결정하게 됩니다. 이것이 바로 MAP의 방법입니다."
   - $$ arg_x max p(x|z) = arg_x max \frac{p(z|x) p(x)}{p(z)} = arg_x max(p(z|x)p(x)) $$
     - 최대 사후 확률을 추정 할 때, 사전 확률도 고려 함. 
     - p(z)는 x에 의한 항이 아니므로 최대값을 찾을 때에는 생략 가능 함.
     - MLE와 차이: p(x)만 추가됨. 즉, 'x의 분포'를 사전에 알고 있다. A 농장의 닭의 무게가 정규 분포 N(1.1kg, 0.1kg^2)을 따른다. 등. 사전 분포 Prior distribution이라고 한다. 사전에 분포 정도는 알고 있을 때, 접근 가능한 방법이다.
       - MAP 문제를 풀라고 할 때에는, 적어도 측정값 z(x+n1, x+n2 등)과, 사전 분포는 넘겨줄 것이다.
         - 사전 분포를 모르면 MLE likelihood로 문제를 풀 수 밖에 없다.
         - 사전 분포가 잘못되었을 수도 있다. 이러면 추정 성능에 악영향을 줄 수 있다.



판서 요약

![MAP]({{site.url}}/images/$(filename)/MAP.png)

로그를 취할 때, -log를 취하면, 앞의 두 항은 MSE, 뒤는 l2 norm이 된다

![MAP2]({{site.url}}/images/$(filename)/MAP2.png)



```python
import numpy as np
array = np.array([1, 3, 2, 4, 5]) # 임의의 배열 생성
max_index = np.argmax(array) # 최대값의 인덱스 찾기
```





사전, 사후? 

: 데이터를 얻기 전 후라는 시간적 의미를 담고 있다



사전 확률: "사전 확률"은 새로운 데이터를 얻기 전에 이미 알고 있는 정보를 확률적으로 표현한 것입니다. 이는 주관적인 믿음이나 이전의 연구 결과, 전문가의 의견 등을 기반으로 할 수 있습니다.



사후 확률: "사후 확률"은 새로운 데이터를 얻은 후에 업데이트된 확률입니다. 사전 확률과 새로운 데이터의 우도(likelihood)를 결합하여 계산됩니다. 이를 통해 초기의 믿음이나 가정을 새로운 데이터에 기반하여 업데이트할 수 있습니다.



이러한 네이밍은 베이즈 통계학의 핵심 개념을 잘 반영하고 있습니다. 즉, 초기에 가지고 있는 정보(사전 확률)를 새로운 데이터가 주어졌을 때 어떻게 업데이트할 것인지(사후 확률)를 설명합니다. 이 과정을 통해 불확실성을 줄이고 더 정확한 추론이나 예측을 할 수 있습니다.



#### 베이즈 정리

![b]({{site.url}}/images/$(filename)/b.png)



#### 베이즈 정리의 유도

![Bayesian_Rule_Induce]({{site.url}}/images/$(filename)/Bayesian_Rule_Induce.png)



#### 베이즈 정리 예시

![Bayesian_Rule_Example_Redball]({{site.url}}/images/$(filename)/Bayesian_Rule_Example_Redball.png)

![Bayesian_Rule_Example_weight]({{site.url}}/images/$(filename)/Bayesian_Rule_Example_weight.png)







![Bayesian_Rule_Example]({{site.url}}/images/$(filename)/Bayesian_Rule_Example.png)





예시 - 언어 모델

- **Likelihood**: 주어진 매개변수(예: 언어 모델의 가중치)가 얼마나 그럴듯한지를 나타내는 확률입니다. 예를 들어, 언어 모델이 "I am" 다음에 "happy"가 올 확률이 얼마나 높은지를 나타냅니다.
- **Prior Distribution**: 매개변수에 대한 사전 지식이나 믿음을 확률 분포로 표현합니다. 예를 들어, 언어 모델의 가중치가 특정 범위 내에 있을 것이라는 믿음을 수학적으로 표현할 수 있습니다.
- **Posterior**: 사전 확률과 우도를 결합하여 업데이트된 확률을 계산합니다. 이 사후 확률을 최대화하는 매개변수 값을 찾는 것이 MAP의 목표입니다.



예시 - 이미지 모델, CNN

- **Likelihood (우도)**: 주어진 이미지 데이터에 대해, 모델의 매개변수(가중치와 편향 등)가 얼마나 그럴듯한 예측을 하는지를 나타냅니다. 예를 들어, 모델이 고양이 이미지를 정확히 '고양이'로 분류할 확률입니다.
  - 우도 계산: 이미지 모델이 주어진 훈련 데이터(이미지와 레이블)에 대해 얼마나 잘 예측하는지를 계산합니다. 예를 들어, 고양이 이미지를 '고양이'로, 개 이미지를 '개'로 잘 분류하는지를 확인합니다.
- **Prior Distribution (사전 확률)**: 매개변수에 대한 사전 지식이나 믿음을 확률 분포로 표현합니다. 예를 들어, 가중치가 너무 크거나 작지 않을 것이라는 믿음을 수학적으로 표현할 수 있습니다.
  - 사전 확률 설정: 모델의 매개변수(가중치)가 특정 범위 내에 있을 것이라는 사전 확률을 설정할 수 있습니다. 이는 모델이 과적합(overfitting)을 피하도록 도와줍니다.
- **Posterior (사후 확률)**: 사전 확률과 우도를 결합하여 업데이트된 확률을 계산합니다. 이 사후 확률을 최대화하는 매개변수 값을 찾는 것이 MAP의 목표입니다.
  - 사후 확률 최대화: 우도와 사전 확률을 결합하여 사후 확률을 계산하고, 이를 최대화하는 매개변수를 찾습니다. 이 과정에서는 보통 베이지안 최적화, MCMC(Markov Chain Monte Carlo) 등의 방법이 사용될 수 있습니다.
  - 모델 업데이트: 사후 확률을 최대화하는 매개변수로 모델을 업데이트합니다. 이렇게 업데이트된 모델은 사전 정보와 새로운 데이터를 모두 고려하여 더 정확한 예측을 할 가능성이 높아집니다.

​	







# 01-21 정보 이론 기초 Entropy, Cross-Entropy, KL-divergence, Mutual information)

#### 소스 코딩이란?

![sourceCoding]({{site.url}}/images/$(filename)/sourceCoding.png)



#### 소스 코딩의 종류

1. **Huffman Coding**: 가장 자주 등장하는 심볼에 더 짧은 코드를 할당하는 방식으로, 텍스트나 이미지 압축에 널리 사용됩니다.
2. **Arithmetic Coding**: 심볼의 확률을 사용하여 하나의 실수로 전체 메시지를 인코딩합니다. 이 방법은 높은 압축률을 제공하지만 계산 복잡성이 높을 수 있습니다.
3. **Run-Length Encoding (RLE)**: 동일한 심볼이 연속으로 등장하는 경우, 그 심볼과 그 길이를 함께 저장합니다. 주로 단순한 그래픽이나 텍스트에 사용됩니다.
4. **Delta Encoding**: 데이터의 차이만을 저장하는 방식으로, 시계열 데이터나 오디오 파일에 주로 사용됩니다.
5. **Lempel-Ziv-Welch (LZW)**: 사전 기반의 압축 방식으로, GIF 이미지나 UNIX의 `compress` 명령어 등에 사용됩니다.
6. **Burrows-Wheeler Transform (BWT)**: 데이터를 더 쉽게 압축할 수 있는 형태로 변환하는 알고리즘으로, bzip2 압축에 사용됩니다.
7. **Golomb Coding**: 일정한 확률 분포를 가진 데이터에 효과적인 압축을 제공합니다. 이는 특히 로스리스 오디오 압축에 유용합니다.
8. **JPEG, MPEG**: 이러한 표준들은 이미지나 비디오 압축에 특화된 여러 가지 소스 코딩 방법을 조합하여 사용합니다. (python library opencv)



#### 머신 러닝에서 자주 사용하는 소스 코딩의 종류

1. **Huffman Coding**: 의사결정 트리(decision tree)나 랜덤 포레스트(random forest)와 같은 트리 기반 모델에서 특성의 중요도를 측정할 때 사용될 수 있습니다. 또한, 텍스트 데이터의 전처리나 압축에서도 사용됩니다. (python library heapq, huffman)
2. **Arithmetic Coding**: 높은 압축률이 필요한 경우, 특히 자연어 처리(NLP)에서 텍스트 데이터를 효율적으로 저장하거나 전송할 때 사용될 수 있습니다. (python library arithmetic-coding)
3. **Delta Encoding**: 시계열 데이터나 연속적인 값을 가진 특성을 처리할 때 사용됩니다. 이 방법은 데이터의 변화를 캡처하여 더 효율적인 표현을 가능하게 합니다. (numpy diff)
4. **Run-Length Encoding (RLE)**: 이미지나 텍스트 데이터에서 반복되는 패턴을 효율적으로 저장할 때 사용될 수 있습니다. 이는 주로 데이터의 전처리 단계에서 적용됩니다. (python library rle, numpy)
5. **Lempel-Ziv-Welch (LZW)**: 텍스트나 시퀀스 데이터를 압축하는 데 사용될 수 있으며, 특히 큰 데이터셋을 다룰 때 유용합니다. (python library lzw)
6. **Golomb Coding**: 이산 확률 분포를 가진 데이터에 효과적인 압축을 제공합니다. 이는 특히 로스리스 오디오 압축이나 특정 유형의 데이터 압축에 사용될 수 있습니다. 



##### 사용 목적

- 데이터의 전처리, 저장, 전송 등을 효율적으로 하기 위해 사용

- 알고리즘의 성능을 향상

- 모델의 크기를 줄임



- Bits 이진수
- Source Coding 소스 코딩: 전송할 데이터를 인코딩한다. 이진수로 효율적으로 표현하자.
- 정보는 랜덤하다.
  - 높은 확률로 나오는 글자는 짧게, 낮은 확률로 나오는 글자는 길게하는 게 효율적 이겠다.
  - 연인과의 카톡에서 하트 ❤️가 나올 확률 P(❤️) = 0.5, ㅗ가 나올 확률 P(ㅗ)=0.001



### Entropy



##### 엔트로피의 의미

![entropy_3]({{site.url}}/images/$(filename)/entropy_3.png)





##### 엔트로피와 소스코딩

![entropy_sourcecoding]({{site.url}}/images/$(filename)/entropy_sourcecoding.png)





##### 엔트로피를 계산하는 목적

![entropy_purpose]({{site.url}}/images/$(filename)/entropy_purpose.png)



##### 강의 내용

❤️를 111, ㅗ를 0로 인코딩 하는 경우

❤️를 0, ㅗ를 111로 인코딩 하는 경우

길이가 훨씬 줄어든다. 효율적이다.

평균 길이를 봐야 한다.

$$ 평균 \sum x_i p_i \rightarrow 문자 길이에 대한 평균 \sum l_i p_i $$

평균 길이는, 문자의 길이 x 문자가 나올 확률, 이 평균 길이가 최소가 되는 것이 좋다.

섀넌 - Entropy가 평균 코드 길이의 이론적 최소임을 밝힘. 하한을 정해줌.



Entropy - 3만 나오는 주사위를 만들었을 때, 엔트로피는 0. 불확실성 없다.

식: $$ \sum_i - p_i log_2 p_i $$ 즉, i 번째 글자에 대해 코드 길이를 $$ p_i $$에 맞춰 $$ -log_2 p_i $$로 하면 된다는 뜻.



![Entropy]({{site.url}}/images/$(filename)/Entropy.png)



### Cross Entropy & KL divergence



강의 메모 - 두서 없음

![cross entropy 2]({{site.url}}/images/$(filename)/cross entropy 2.png)





#### Cross Entropy

$$ \sum_i - p_i log_2 q_i $$



$$ p_i $$는 실제 값을 모르거나, 문자길이가 소수일 수 없으므로 대신 $$ q_i $$를 넣기도 하는데, 이를 통해 구한 $$ \sum_i - p_i log_2 q_i $$를 cross entropy라고 하며, cross entropy - entropy를 KL divergence라고 한다.



##### Cross Entropy를 사용하는 목적



![cross_entropy3]({{site.url}}/images/$(filename)/cross_entropy3.png)





#### KL divergence

##### KL divergence

![KL_divergence]({{site.url}}/images/$(filename)/KL_divergence.png)



$$ \sum_i - p_i log_2 \frac{p_i}{q_i} $$  



cross entropy는 entropy보다 항상 크다. 빼면 양수 값이다.



p와 q의 분포 차이(거리)로 해석 가능



만약 출력 q_i 만 업데이트할 수 있다면, cross entropy를 줄이는거나 KL을 을 최소화하는 해는 동일하다.





### Mutual Information



![mutual information]({{site.url}}/images/$(filename)/mutual information.png)





> 참고. 손실 함수

### 회귀 문제 (Regression)

1. **Mean Squared Error (MSE)**: 예측값과 실제값의 차이의 제곱의 평균입니다.
2. **Mean Absolute Error (MAE)**: 예측값과 실제값의 차이의 절대값의 평균입니다.
3. **Huber Loss**: MSE와 MAE를 결합한 손실 함수로, 오차가 작을 때는 MSE를, 큰 경우에는 MAE를 사용합니다.

### 분류 문제 (Classification)

1. **Cross-Entropy Loss**: 두 확률 분포 사이의 차이를 측정합니다. 이진 분류와 다중 클래스 분류에 모두 사용됩니다.
2. **Hinge Loss**: 서포트 벡터 머신(SVM)에서 사용되며, 마진을 최대화하는 방향으로 모델을 학습시킵니다.
3. **Zero-One Loss**: 예측이 정확한 경우 0, 그렇지 않은 경우 1을 반환합니다. 이는 이론적으로는 유용하지만, 미분이 불가능하기 때문에 실제로는 잘 사용되지 않습니다.

### 순서형 문제 (Ranking)

1. **RankNet Loss**: 순서를 예측하는 문제에서 사용되며, 주로 검색 엔진 최적화에 사용됩니다.
2. **Pairwise Loss**: 두 개의 아이템이 주어졌을 때, 어느 것이 더 높은 순위인지를 예측하는 문제에 사용됩니다.

### 시퀀스 문제 (Sequence)

1. **Sequence Loss**: 시퀀스 데이터를 다룰 때 사용되며, RNN, LSTM, GRU 등에서 활용됩니다.
2. **CTC (Connectionist Temporal Classification) Loss**: 시퀀스 레이블링 문제에서 사용되며, 특히 음성 인식이나 핸드라이팅 인식에 사용됩니다.

### 생성 모델 (Generative Models)

1. **Generative Adversarial Loss**: GANs에서 사용되며, 생성자와 판별자 사이의 손실을 최소화합니다.
2. **Kullback-Leibler Divergence**: 두 확률 분포의 차이를 측정하는 데 사용됩니다.

### 기타

1. **Cosine Similarity Loss**: 두 벡터 사이의 코사인 유사도를 최대화하는 방향으로 모델을 학습시킵니다.
2. **Triplet Margin Loss**: 유사한 아이템은 가깝게, 다른 아이템은 멀게 배치하는 것을 목표로 합니다.
