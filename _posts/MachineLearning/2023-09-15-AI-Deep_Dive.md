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

*AU Deeo Duve Note*


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

![scalar_vector_diff_2png]({{site.url}} /images/$(filename)/scalar_vector_diff_2png.png)

![scalar_vector_diff_3]({{site.url}}/images/$(filename)/scalar_vector_diff_3.png)

![scalar_vector_diff_4]({{site.url}}/images/$(filename)/scalar_vector_diff_4.png)





# 01-11 왜 그라디언트는 가장 가파른 방향을 향할까





- Loss 함수 L(w)를 $$ w = w_k $$에서 1차까지만 Taylor series를 전개하면 (k위치에서)

 

$$ L(w) = C_0 + C_1(w_1 - W_{k_1, 1}) + C_2(w_2 - w_{k_1, 2})$$

$$ = C_0 + [w_1 - w_{k_1 1}, w_2 - w_{k_1 2} ] 

\begin{bmatrix}
C_1  \\
C_2
\end{bmatrix} $$





![gradient_stiff]({{site.url}}/images/$(filename)/gradient_stiff.png)

 그라디언트 방향으로 업데이트를 하면, L을 가장 크게 바꿀 수 있는 방향이다.



learning rate가 필요한 이유: 근사를 활용하기 때문에 너무 많이 움직이면 근사가 깨짐



![gradient_stiff_2]({{site.url}}/images/$(filename)/gradient_stiff_2.png)



테일러 급수를 이용한 증명

![gradient_stiff_3]({{site.url}}/images/$(filename)/gradient_stiff_3.png)







# 01-12 벡터를 벡터로 미분하는 법



- back propagation을 간단히 표현하는 데에 도움





# 01-13 벡터를 벡터로 미분할 때의 연쇄 법칙

# 01-14 스칼라를 행렬로 미분하는 법

# 01-15 행렬을 행렬로, 벡터를 행렬로 미분하는 법

# 01-16 랜덤 변수와 확률 분포

# 01-17 평균과 분산

# 01-18 균등 분포와 정규 분포

# 01-19 최대 우도 추정 (MLE)

# 01-20 최대 사후 확률 (MAP)

# 01-21 정보 이론 기초 Entropy, Cross-Entropy, KL-divergence, Mutual information)

