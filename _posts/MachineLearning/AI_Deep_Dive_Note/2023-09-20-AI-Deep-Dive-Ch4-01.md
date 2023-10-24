---
layout: single
title: "AI Deep Dive, Chapter 4. 딥러닝, 그것이 알고싶다 01. MLP, 행렬과 벡터로 나타내기 & 왜 non-linear activation이 중요할까"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 4 - 01. MLP 행렬과 벡터로 나타내기 & 왜 non-linear activation이 중요할까



- example MLP

  - hidden layer: 1개

  - input layer: $$ x_1, x_2 $$

  - activation functions: $$ f_1, f_2 $$

  - f1에 들어가는 입력을 정리하면(복잡) 

    ​	$$ x_1 w_1 + x_2 w_2 + b + ... $$

  - 행렬로 표현하면(간결)

    $$

    f1 \left ( 

    \begin{bmatrix}
     x_1 & x_2 
    \end{bmatrix}

    \begin{bmatrix}
     w_1 & w_3 & w_5 \\ 
     w_2 & w_4 & w_6
    \end{bmatrix} 

    + \begin{bmatrix}
    b_1 & b_2 & b_3
    \end{bmatrix} 

    \right )

    $$	 

  - 벡터로 표현하면(더 간결)

    
    $$
    \underline{f_1}\left ( \underline{x}*  \underline{W_1} + \underline{b_1} \right )
    $$

    - dimension check
      - 1x2 * 2x3 + 1x3 -> 1x3

  - 그 다음 레이어 추가로 표현

    
    $$
    \underline{f_2} \left ( \underline{f_1}\left ( \underline{x}*  \underline{W_1} + \underline{b_1} \right ) * \underline{W_2} + \underline{b_2} \right )
    $$

    - dimension check
      - 3x2 + 1x2 -> 1x2

  - MLP는 weight 행렬 곱하고 bias 행렬 더하여 activation function에 다 넣어주면 됨. fully connected layer라서, 간결하게 표현이 가능함.

![ch0401_1]({{site.url}}/images/$(filename)/ch0401_1.png)





![ch0401_2]({{site.url}}/images/$(filename)/ch0401_2.png)



- 깊게 표현할 때에, linear activation은 말짱 헛거임.
- 분배법칙으로 전개해보면, 결국 하나의 layer로 치환가능해짐.

![ch0401_3]({{site.url}}/images/$(filename)/ch0401_3.png)



![ch0401_4]({{site.url}}/images/$(filename)/ch0401_4.png)



![ch0401_5]({{site.url}}/images/$(filename)/ch0401_5.png)









# 추가 조사

##### linear activation으로 MLP를 구성하면 표현력에 한계가 생긴다?

선형 활성화 함수(Linear Activation)를 사용하여 다층 퍼셉트론(MLP)을 구성하면 표현력에 한계가 생길 수 있습니다. 이러한 이유로 선형 활성화 함수는 일반적으로 머신 러닝 모델에서 사용되지 않거나 제한적으로 사용됩니다.

선형 활성화 함수인 f(x) = x는 입력에 대한 선형 변환만 수행하며, 이것은 층을 쌓더라도 결국 하나의 선형 변환으로 표현되기 때문에 깊은 신경망에서의 표현 능력이 제한됩니다. 따라서 MLP의 모든 층이 선형 활성화 함수를 사용한다면 단일 퍼셉트론과 동일한 표현 능력을 가지게 됩니다.

따라서 신경망의 표현력을 높이기 위해서는 비선형 활성화 함수를 사용해야 합니다. 주로 사용되는 활성화 함수 중 하나는 시그모이드 함수나 하이퍼볼릭 탄젠트 함수와 같은 S자 형태의 비선형 함수입니다. 또한, ReLU(Rectified Linear Unit)와 그 변형(Leaky ReLU, ELU 등)도 널리 사용됩니다.

이러한 비선형 활성화 함수는 모델이 비선형 데이터 패턴을 학습하고 다양한 표현을 추출할 수 있게 해줍니다. 따라서 MLP에서는 주로 비선형 활성화 함수를 사용하여 표현력을 향상시킵니다.

