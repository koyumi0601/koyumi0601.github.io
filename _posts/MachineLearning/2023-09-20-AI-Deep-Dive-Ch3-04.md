---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 04. 가중치 초기화 기법 정리"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 04. 가중치 초기화 기법 정리

# 강의

- 랜덤하게, 0 근처로 잡아라

  - 충분히 안정적으로 수렴한다.

- LeCun

  - ![ch0304_1]({{site.url}}/images/$(filename)/ch0304_1.png)
  - 평균 0, 분산 $$ \frac{1}{N_in} $$ 으로, weight들을 초기화해라
    - 입력 노드가 3개이면 1/3

  - uniform하게 잡든, normal distribution으로 잡든

- Xavier

  - sigmoid, tanh 사용하는 신경망

    ![ch0304_2]({{site.url}}/images/$(filename)/ch0304_2.png)

- He

  - ReLU 사용하는 신경망

    ![ch0304_3]({{site.url}}/images/$(filename)/ch0304_3.png)



- Q) zero initialization 혹은 전부 1로 초기화하면 문제점?
  - back propagation에서 문제가 생긴다!
  - 왜냐하면 모든 가중치가 같은 값을 가지기 때문에 *역전파* 과정에서 *그래디언트가 동일하게 업데이트*되어 네트워크가 *학습하는 데 문제가 발생*할 수 있습니다.





# 실제 코드



```python
import tensorflow as tf

# lecun_initializer = tf.keras.initializers.lecun_normal() # LeCun 초기화
initializer = tf.initializers.glorot_uniform() # Xavier 초기화
# he_initializer = tf.keras.initializers.he_uniform() # He 초기화

# 가중치 변수 생성 및 초기화
weight = tf.Variable(initializer(shape=(input_dim, output_dim)))
```



```python
import torch
import torch.nn.init as init

weight = torch.empty(input_dim, output_dim)

# init.normal_(weight, mean=0, std=1) # LeCun 초기화
init.xavier_uniform_(weight) # Xavier 초기화
# init.kaiming_uniform_(weight, mode='fan_in', nonlinearity='relu') # He 초기화

# 파이토치의 경우 직접 가중치를 생성한 후 초기화 함수를 호출하여 초기화합니다.
```



# 설명



> **1. LeCun 초기화: **<br>
>
> LeCun 초기화는 시그모이드 활성화 함수와 함께 사용하기 위해 개발되었습니다. 이 초기화 방법은 가중치를 작은 무작위 값으로 초기화하며, 평균(mean)이 0이고 표준 편차(std)가 1인 가우시안 분포 또는 -1에서 1 사이의 균일 분포를 사용합니다. 주로 시그모이드 함수와 하이퍼볼릭 탄젠트 함수와 함께 사용됩니다. <br>
>
> **2. Xavier 초기화 (Glorot 초기화):** <br>
>
> Xavier 초기화는 시그모이드 및 하이퍼볼릭 탄젠트 활성화 함수와 함께 사용하기 위해 개발되었습니다. 이 초기화 방법은 가중치를 입력 차원과 출력 차원에 따라 조정된 작은 무작위 값으로 초기화합니다. 평균(mean)이 0이고 표준 편차(std)가 <br>
>
> sqrt(2 / (입력 차원 + 출력 차원)) <br>
>
> 로 설정된 가우시안 분포 또는 -sqrt(6 / (입력 차원 + 출력 차원))에서 sqrt(6 / (입력 차원 + 출력 차원)) 사이의 균일 분포를 사용합니다. <br>
>
> **3. He 초기화:** <br>
>
> He 초기화는 ReLU (Rectified Linear Unit) 활성화 함수와 함께 사용하기 위해 개발되었습니다. ReLU는 입력이 양수인 경우에는 선형 변환을 수행하므로, 가중치 초기화를 더 큰 값으로 설정하여 효과적인 학습을 도와줍니다. He 초기화는 평균(mean)이 0이고 표준 편차(std)가 sqrt(2 / 입력 차원)인 가우시안 분포 또는 -sqrt(6 / 입력 차원)에서 sqrt(6 / 입력 차원) 사이의 균일 분포를 사용합니다. <br>
>
> **어떤 초기화 방법을 사용해야 할까요? **<br>
>
> - **시그모이드 또는 하이퍼볼릭 탄젠트 활성화 함수를 사용하는 경우:** Xavier 초기화를 사용하는 것이 일반적으로 좋습니다. Xavier 초기화는 이러한 활성화 함수와 잘 맞아서 학습이 안정적으로 수행됩니다. <br>
> - **ReLU 활성화 함수를 사용하는 경우:** He 초기화를 사용하는 것이 좋습니다. He 초기화는 ReLU 함수와 잘 어울리며, 그라디언트 소실 문제를 일부 해결해줍니다. <br>
>
> 주의할 점은 초기화 방법의 선택은 모델의 구조와 사용되는 활성화 함수에 따라 다를 수 있으며, 실험을 통해 최적의 초기화 방법을 찾는 것이 좋습니다. 또한 가중치 초기화는 하이퍼파라미터 중 하나이므로, 모델의 성능을 향상시키기 위해 조정해야 할 수 있습니다. <br>