---
layout: single
title: "AI Deep Dive, Chapter 7. 깊은 인공신경망의 고질적 문제와 해결 방안 02. vanishing gradient의 해결방안 ReLU"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 7 - 02. vanishing gradient의 해결방안 ReLU

- vanishing gradient의 주범: sigmoid

- 미분값이 그리 작지 않은 함수를 activation으로 써보자, ReLU

- ReLU Rectified Linear Unit

  - 적어도 양수쪽으로 인풋이 들어오면, 적어도 미분 값이 1은 됨. 
    - 적어도 activation에 의해서 (액웨액웨액웨...액앤)에서 0으로 가는 일은 없겠다. 

    - 다만 마이너스 인풋이면 0이 되긴 하는데, 아직 bias 남아있으니까 아예 다 없어지는 건 아니지만.. 
      - 그래도 음수쪽을 살짝 살려줌 
        - Leaky ReLU: y = 0.01x
        - 학습하자, Parametric ReLU: y = ax
          - 미분은 되나? chain rule에 의해서 미분만 되면, 학습 가능


  
