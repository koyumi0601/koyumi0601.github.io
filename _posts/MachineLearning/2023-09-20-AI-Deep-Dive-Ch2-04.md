---
layout: single
title: "AI Deep Dive, Chapter 2. 왜 현재 AI가 가장 핫할까? 04. 자기지도학습"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 2 - 04. 자기지도학습

*Doersch, Carl, Abhinav Gupta, and Alexei A. Efros. "Unsupervised visual representation learning by context prediction." Proceedings of the IEEE international conference on computer vision. 2015*



데이터가 많을수록 무조건 좋은데, 정답을 알고 있는 데이터가 너무 적다.



정답을 만드는 데 비용이 너무 많이 든다. 준지도학습도 마찬가지.



자기지도학습은 진짜 풀려고 했던 문제 말고 다른 문제를 새롭게 정의해서 먼저 풀어본다



![자기지도학습]({{site.url}}/images/$(filename)/자기지도학습.png)

정답은 고양이가 정답임.

정답은 라벨링되어 있지 않음

파란색(패치)의 위치를 랜덤하게 잡는다. 사이즈를 정해서..

패치의 이미지와, 주변의 상대적인 위치 1, 2, 3, 4, 5, 6, 7, 8 정의를 해놓고, 아무거나 똑 떼서, 두장의 사진을 입력하고, 출력은 상대적인 위치 값이 튀어나오도록, 학습을 시킨다.

이 사진이 고양이인지, 강아지인지는 알 필요가 없다. 파란 패치 + 주변부 이미지 두 장의 사진을 보여주고, 주변 위치 값이 나오도록 출력하는 것을 먼저 풀어본다.

![blue_patch]({{site.url}}/images/$(filename)/blue_patch.png)

- 1 이미지 , blue 패치 이미지 를 입력으로 넣으면, 상대 위치 1이 나오도록 학습 시킨다.
- 미리 푸는 문제 
- 강아지 고양이 분류 문제와는 다른 문제이다. 그런데, 분류 문제를 푸는데에 일정 정도 도움을 주는 문제이다. 이미지가 상대적인 위치로 이루어져 있음을 학습한다.
- 이후, 적은 정답지로 학습을 하면 좀 낫다.
- 데이터 안에서 self로 정답(label)을 만듦 -> 새롭게 정의한 문제에 대한 정답 -> 그래서 이름이 자기지도 학습



순서

- pretext task 학습으로 pre-training -> pre-trained model (블루 패치와 주변부 이미지를 넣으면 주변부 이미지의 번호를 출력하도록 학습. 분류문제가 아님)
- downstream task(분류)를 풀기 위해 transfer learning함 (정답지가 있는 것으로 지도학습) -> pre-trained model의 출력과 강아지/고양이 분류 문제는 출력이 다르므로, 출력층을 살짝 바꿔준다.
- 레이블이 되어 있는 데이터로 학습을 시키면, classification model을 만들 수 있다.





- 지도 학습은 두 번째 단계만 있었던 것.

- 자기주도학습은 1번을 먼저 풀고, 2번을 풀게 하는 것.
- 왜 하냐? 정답 데이터가 적어서





I am an instructor.

영어를 한글로 바꾸는 모델을 만들 때, 



-> 나는 강사입니다.



정답을 모르는 상황



자기지도학습으로 다른 걸 풀어봄.



I (am) an instructor. am을 지우고 am이 튀어나오도록 학습시킴. 문맥을 이해하게 됨.



그 이후 다른 문장, 정답 문장을 가지고 학습을 하면 훨씬 번역을 잘 한다.



본 논문에서는 pretext task를 context prediction으로 제안



이미지에도 맥락이 있다



요즘에 많이 쓰는 자기지도 학습으로는 Pretext task로 Contrastive learning을 많이 쓴다.

![pretext]({{site.url}}/images/$(filename)/pretext.png)

Chen, Ting, et al. "A simple framework for contrastive learning of visual representations. " International conference on machine learning. PMLR, 2020



CNN 모델

(강아지 사진) - 정답은 모름 - 일부를 떼서 두장을 만듦. 두 패치는 출처는 같다. 각각 CNN모델을 통과시킨다.

결과물 1, 결과물 2

출처가 같은 두 이미지를 넣은 것이라면 두 결과물은 출력값이 비슷해야 한다. -> 그렇게 만들자 

![pretext_dog_cnn]({{site.url}}/images/$(filename)/pretext_dog_cnn.png)



그 다음에는 다른 사진(의자)인지를 안다면, 3, 4끼리는 출력값이 비슷해야 한다.

1,2와 3,4는 결과가 멀어지도록 (리펠, 밀어낸다.) 3,4끼리는 attract 당겨야 한다.

![pretext_dog_cnn_chair]({{site.url}}/images/$(filename)/pretext_dog_cnn_chair.png)

색깔이 달라지는 것은 augmentation이라고해서, 약간의 변형을 주는 것. 그래도 출처는 같다. 
