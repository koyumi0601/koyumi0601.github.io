---
layout: single
title: "AI Deep Dive, Chapter 4. 딥러닝, 그것이 알고싶다 03. 행렬미분을 이용한 Backpropagation"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 4 - 03. 행렬미분을 이용한 Backpropagation

들어가기에 앞서, 딥러닝을 위한 필수 기초 수학(표현법)은 아래와 같다.



##### 스칼라를 벡터로 미분하는 법

$$ df = \frac{\partial{f}}{\partial{x_1}}dx_1 + \frac{\partial{f}}{\partial{x_2}}dx_2 = 

\begin{bmatrix}
 dx_1&dx_2 
\end{bmatrix}

\begin{bmatrix}
 \frac{\partial{f}}{\partial{x_1}} \\

\frac{\partial{f}}{\partial{x_2}}
\end{bmatrix}

= d\underline{x} \frac{\partial{f}}{\partial{x}^T}

 $$



##### 벡터를 벡터로 미분하는 법

$$ d\underline{f} 

= \begin{bmatrix}
 df_1&df_2 
\end{bmatrix}

= \begin{bmatrix}
 dx_1&dx_2 
\end{bmatrix} 

\begin{bmatrix}
 \frac{\partial{f_1}}{\partial{x_1}} & \frac{\partial{f_2}}{\partial{x_1}} \\

\frac{\partial{f_1}}{\partial{x_2}} & \frac{\partial{f_2}}{\partial{x_2}}
\end{bmatrix}

= d\underline{x} \frac{\partial{\underline{f}}}{\partial{\underline{x}^{T}}}

$$

예시로, 

$$ y = f(x) = xA  $$를 x로 미분

$$ d\underline{f} = d\underline{x}  \underline{A}  $$ 이므로 

$$ \frac{\partial{f}}{\partial{x}^{T}} = \underline{A} $$로 바로 나온다

*f의 변화량을 구해서, dx 옆에 있는 게 미분 값이다!*



##### 행렬을 행렬로 미분하는 법



- vectorize 하고 벡터를 벡터로 미분하는 방법을 취한다
  - $$ d vec(F) = d vec(x) \frac{ \partial{vec(F)}}{ \partial{vec^T(x)}} $$
  - 변화량을 구해보고, $$ d vec(x) $$ 옆에 있는 거가 미분 값이다.

- [kronecker product](https://freshrimpsushi.github.io/posts/kroneker-product-of-matrices/) 표기를 이용해서 간략히 표기한다

- [행렬 미분](https://geniewishescometrue.tistory.com/entry/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98%ED%95%99-%ED%96%89%EB%A0%AC%EB%AF%B8%EB%B6%84-Matrix-Calculus)

![ch0403_7]({{site.url}}/images/$(filename)/ch0403_7.png)



이런 기본 아이디어를 가지고, $$ \underline{y} = \underline{x} W $$를 미분해보자

1. (편법) 행렬을 벡터로 바꾼다: vectorize
   - Y = XA
   - vec(X)
   - vec(Y)
2. 벡터를 벡터로 미분한다
   - $$ \frac{\partial {\underline{y}}}{\partial {\underline{x^T}}} $$
3. 아래의 d vec(F) = Y = XA 이다. 이건 예시로 단순한 신경망

![ch0403_6]({{site.url}}/images/$(filename)/ch0403_6.png)



#### 본격적으로

위를 이용해서, Loss를 weight로 편미분 해보자



![ch0403_8]({{site.url}}/images/$(filename)/ch0403_8.png)

MLP를 벡터와 행렬을 이용해서 나타내보았다.

$$ d_1 = n_0 W_1 + b_1 => n_1 = f_1 (d_1) => $$

$$ d_2 = n_1 W_2 + b_2 => n_2 = f_2 (d_2) => L = (n_2 - y)(n_2 - y)^T $$

이전 챕터에서, 도로 스칼라로 미분하는 것이 아쉬워서, W(3x2)행렬로 미분하는 것으로 다시 설명해보고자 한다.



각 입력($$n_0 $$), 들어가는 값 $$d_1 $$ 등을 벡터로 간략히 표기한 후, 행렬과 벡터식으로 나타내보자.

![ch0403_9]({{site.url}}/images/$(filename)/ch0403_9.png)

훨씬 간결하다

이 상황에서, weight 행렬 W로 한방에 미분하면 훨씬 간결할 것이다.

이때 Loss함수도 행렬과 벡터로 표현하자면 아래와 같다.

![ch0403_11]({{site.url}}/images/$(filename)/ch0403_11.png)

위에 썼던 이 수식이다. $$ L = (n_2 - y)(n_2 - y)^T $$



이번엔 vectorize한 후 신경망 모양대로, chain rule을 적용해보자

![ch0403_12]({{site.url}}/images/$(filename)/ch0403_12.png)





![ch0403_2]({{site.url}}/images/$(filename)/ch0403_2.png)





먼저, w2로 loss를 미분한 것을 구해보자

![ch0403_20]({{site.url}}/images/$(filename)/ch0403_20.png)



(빨간색) $$ \frac{\partial{L}}{\partial{n_2}^T} $$를 구해보자

변화량 dL을 유도한다

![ch0403_16]({{site.url}}/images/$(filename)/ch0403_16.png)



(초록색)을 구해보자

주대각 성분을 제외한 n, d들은 서로 영향이 없으므로 0을 채울 수 있다.

$$ diag(f'_2(d_2)) $$라고 줄여서 계산할 수 있다.

그러니까 $$ f'_2() $$를 구하면된다. 즉, f2를 미분하면 된다.

다만 미분 값이 여기에 올라가 있으면 된다. 그걸 표기를 diag(~)라고 하면 된다.

![ch0403_17]({{site.url}}/images/$(filename)/ch0403_17.png)



(분홍색) $$ \frac {\partial{d_2}} { \partial{W_2^T}}  $$ 을 구해보자 

![ch0403_19]({{site.url}}/images/$(filename)/ch0403_19.png)

기초수학: 벡터를 행렬로 미분하는 파트 참고

![ch0403_18]({{site.url}}/images/$(filename)/ch0403_18.png)



W1에 대해서도 전개해보자

![ch0403_22]({{site.url}}/images/$(filename)/ch0403_22.png)

위와 같은 방식으로 써주면 된다

(빨간색)

![ch0403_23]({{site.url}}/images/$(filename)/ch0403_23.png)

(초록색)

![ch0403_24]({{site.url}}/images/$(filename)/ch0403_24.png)

(파란색)

![ch0403_25]({{site.url}}/images/$(filename)/ch0403_25.png)

(노란색)

![ch0403_26]({{site.url}}/images/$(filename)/ch0403_26.png)

(분홍색)

![ch0403_27]({{site.url}}/images/$(filename)/ch0403_27.png)

형태: 액-웨-액-웨-액-공통





끝이긴 한데, 좀만 더 전개하면, 스칼라를 행렬로 미분 $$ \frac{\partial{L}} { \partial{ \underline{W_1}}} $$으로 구할 수 있다.



위는 행렬로 미분한 건 아니고 벡터라이즈해서 미분한 상태임

= 세로로 쌓아놓은 상태

다시 뒤집어서 원래 행렬 형태로 나타낼 수 없을까?

![ch0403_28]({{site.url}}/images/$(filename)/ch0403_28.png)

![ch0403_29]({{site.url}}/images/$(filename)/ch0403_29.png)

