---
layout: single
title: "[Study Summary] Activation Functions"
categories: machinelearning
tags: [ML, Machine Learning, AI, Summary]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true



---

*Activation Functions for machine learning*

강의 노트가 아니라 정보만 정리한 것

![activation_fun]({{site.url}}\images\2023-12-13-ActivationFunctions\activation_fun.png)





1. **이미지 분류 (Image Classification)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 이미지 분류 작업에서는 주로 ReLU, Leaky ReLU, Swish와 같은 활성화 함수가 사용됩니다. 이들은 이미지 분류 문제에서 잘 동작하며, 합성곱 신경망 (CNN)과 함께 주로 사용됩니다.
2. **이미지 분할 (Image Segmentation)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 이미지 분할은 객체의 경계를 감지하는 작업으로, 이미지 분류와 비슷한 활성화 함수를 사용합니다. 주로 CNN과 함께 사용됩니다.
3. **객체 검출 (Object Detection)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 객체 검출은 이미지에서 객체의 위치와 경계 상자를 예측하는 작업입니다. 이미지 분류와 유사한 활성화 함수가 주로 사용됩니다.
4. **이미지 캡셔닝 (Image Captioning)**:
   - 주로 사용되는 활성화 함수: LSTM, GRU, Swish
   - 설명: 이미지 캡셔닝은 이미지에 대한 설명을 생성하는 작업으로, 순환 신경망 (RNN)과 함께 LSTM, GRU와 같은 활성화 함수가 사용됩니다.
5. **비주얼 임베딩 (Visual Embedding)**:
   - 주로 사용되는 활성화 함수: ReLU, Swish
   - 설명: 비주얼 임베딩은 이미지를 고정 차원 벡터로 변환하는 작업으로, ReLU 및 Swish와 같은 활성화 함수를 사용한 신경망이 주로 활용됩니다.
6. **생성 모델 (Generative Models)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish (생성자), Sigmoid (판별자)
   - 설명: 생성 모델은 이미지 생성 작업에 사용되며, 생성자 네트워크에서는 ReLU, Leaky ReLU, Swish와 같은 활성화 함수가, 판별자 네트워크에서는 Sigmoid가 주로 사용됩니다.



```py
import matplotlib.pyplot as plt
import numpy as np

# 활성화 함수 그래프 그리기 함수
def plot_activation_functions():
    x = np.linspace(-5, 5, 1000)
    
    # 활성화 함수 정의
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    swish = x * sigmoid
    hard_sigmoid = np.clip(0.2 * x + 0.5, 0, 1)
    hard_swish = x * hard_sigmoid
    elu = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    # 활성화 함수와 특장점 및 단점을 포함한 정의
    # Activation functions with their descriptions
    activation_functions = [
        (sigmoid, 'Sigmoid', '$\\frac{1}{1 + e^{-x}}$', 'Advantages: Bounded between 0 and 1\nDrawbacks: Gradient vanishing'),
        (tanh, 'Tanh', '$\\tanh(x)$', 'Advantages: Bounded between -1 and 1\nDrawbacks: Gradient vanishing'),
        (relu, 'ReLU', '$\\max(0, x)$', 'Advantages: Fast computation, easy optimization\nDrawbacks: Dead neurons'),
        (leaky_relu, 'Leaky ReLU', '$x$ if $x > 0$ else $0.01x$', 'Advantages: Mitigates dead neurons\nDrawbacks: Varying performance'),
        (swish, 'Swish', '$x \\cdot \\frac{1}{1 + e^{-x}}$', 'Advantages: Smooth non-linearity\nDrawbacks: Slow computation'),
        (hard_sigmoid, 'Hard Sigmoid', '$0$ if $x < -2.5$ else $0.2x + 0.5$ if $-2.5 \\leq x \\leq 2.5$ else $1$', 'Advantages: Fast, simple computation\nDrawbacks: Reduced accuracy'),
        (hard_swish, 'Hard Swish', '$x \\cdot (1 / (1 + e^{-x}))$', 'Advantages: Easily optimized\nDrawbacks: Reduced accuracy'),
        (elu, 'ELU', '$x$ if $x > 0$ else $(e^x - 1)$', 'Advantages: Mitigates dead neurons, smooth non-linearity\nDrawbacks: Slower computation'),
        (gelu, 'GELU', '$0.5x (1 + tanh(sqrt(2/π)(x + 0.044715 * x^3)))$', 'Advantages: Smooth non-linearity\nDrawbacks: Slower computation')
    ]
    

    # 유사한 도메인에서 활성화 함수 색상 설정
    colors = ['b', 'b', 'g', 'g', 'r', 'c', 'r', 'c', 'r']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 더 큰 크기로 조절
    plt.subplots_adjust(wspace=0.4)  # 서브플롯 간격 조정
    for i, (activation_function, function_name, equation, description) in enumerate(activation_functions):
        ax = axes[i // 3, i % 3]
        y = activation_function
        ax.plot(x, y, label=f"{equation}", color=colors[i])
        ax.legend(loc='upper left')
        ax.grid(True)
        # ax.text(0.5, -0.25, description, transform=ax.transAxes, fontsize=10, va='top')
        # ax.text(0.5, 1.1, description, transform=ax.transAxes, fontsize=10, va='center', ha='center')
        ax.text(0.5, -0.25, description, transform=ax.transAxes, fontsize=10, va='bottom', ha='center')
        
        # 각 subplot의 타이틀에 활성화 함수 이름 설정
        ax.set_title(function_name)
        
    # plt.tight_layout()
    plt.tight_layout(pad=2.0)
    # plt.subplots_adjust(wspace=0.4, hspace=0.6)  # wspace와 hspace 값을 조정하여 간격 설정
    plt.show()

# 모든 활성화 함수 그래프 그리기
plot_activation_functions()
```

