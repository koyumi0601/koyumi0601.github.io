---
layout: single
title: "ImageNet"
categories: machinelearning
tags: [ML, Machine Learning, AI, Legend13]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*Legend 13 (Image)  Voice Transciption*

# 강의록

![ImageNet]({{site.url}}\images\2023-10-25-Day1-ImageNet\ImageNet.png)





# Keyword and Summary

- ImageNet Challenge
  - 대규모 이미지 데이터셋을 사용하는 대회
  - image classification
- 데이터
  - 클래스 1000개
  - 128만장
  - 다양한 사이즈 (resize, crop 필요)
- 일반적인 모델의 구조
  - last node: MLP(flatten -> fully connected)
- Top 5 Error
  - 모델이 예측한 상위 5개 카테고리 중에 정답이 있는 경우를 맞다고 처리하고, 그렇지 않으면 틀린 것으로 처리하는 에러 측정 지표
- 전망
  - 2014년 이후 AI가 인간을 뛰어넘음.




# Transcription
- 클로바노트 https://clovanote.naver.com/

##### 받아쓰기

참석자 1 00:09
이미지넷 챌린지는 이렇게 수많은 이미지 128만 장의 이미지를 가지고 1000종 분류하는 분들 1000종 그럼 맨 마지막 노드의 개수로 적절한 것은 몇 개일까요? 천개죠. MLP 통과하고 나서 맨 마지막은 무조건 1000개로 끝납니다. 

제가 보여드릴 모든 모델들이 맨 마지막에는 Fully Connected를 쓰고요. 1000개로 노드가 끝이 나요. 

그렇습니다. 그래서 2012년부터 딥러닝이 휩쓸었죠. 



top 5 에러율이라고 해서 top 5 분류 안에 정답이 있으면 정답 처리 아니면 에러 처리를 하는 겁니다. 
그래서 예측 5개 중에 이제 좀 봐주는 거예요. 좀 봐주는 거 예측을 5개 5순위까지 하게 하고 그중에 정답이 있으면 그래 니 맞았어 이렇게 해주는 거고 만약에 그 안에 아예 없으면 틀렸어. 이렇게 해주는 게 top 5 에러인 겁니다. 



이미지 사이즈가 굉장히 다양해요. 생각보다. 그래서 resize, crop 이런 걸 잘 해줘야 됩니다. 



현대에 와서는 2015년 이후로는 사람이 감히 ai한테 대적할 수가 없게 됐어요. 이미지 인식에 있어서는 사람이 이길 수가 없게 됐고 한 10년 정도만 지나면 사람이 감히 어떻게 운전을 하냐 ai한테 맡겨야지 안전 운전해야지 이렇게 되는 시대가 오지 않을까요? 지금은 ai 못 믿고 하지만 마찬가지예요.  2014년에도 마찬가지였어요. 사람이 분류를 해야지 어떻게 ai 따위한테 분류를 맡기냐 이제는 그런 말 안 하잖아요. 10년 뒤에는 자율주행에 있어서도 똑같이 그런 말 하지 않을까 왜냐하면 그것도 비전으로 하는 거니까 비전 데이터로 하는 거니까 



오케이 됐습니다. 여기까지 일단 질문 있으신가요? 버튼 한번 눌러주십시오.


오케이 좋습니다.  그러면 10분 쉬었다가 9시에 돌아와서 VGGNet 한번 살펴보도록 하겠습니다. 수고하셨습니다.