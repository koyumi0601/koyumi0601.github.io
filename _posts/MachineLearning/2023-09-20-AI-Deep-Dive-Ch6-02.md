---
layout: single
title: "AI Deep Dive, Chapter 6. 인공신경망 그 한계는 어디까지인가? 02. Universal Approximization Theorem (실습을 통한 확인)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 6 - 02. Universal Approximization Theorem (실습을 통한 확인)



- 구글 코랩 검색 [https://colab.research.google.com/?hl=ko](https://colab.research.google.com/?hl=ko)
- 새노트
- 구글 드라이브 [https://drive.google.com/drive/my-drive?hl=ko](https://drive.google.com/drive/my-drive?hl=ko)
  - 내 드라이브 > CoLab Notebooks
  - 배포자료 붙여넣기

- shift enter: 셀 실행 후 다음 셀로 이동
- control enter: 셀 실행



##### 구성

- import
- hyperparameter 설정
- model 설정
- 학습 (반복)
- 검증 loss



##### 첨부 코드 확인 요망







##### 참고

- .pt 파일 

  - PyTorch에서 모델의 가중치나 전체 모델을 저장할 때 사용하는 파일 확장자

  - torch.save / torch.load

  - 모델의 가중치만 저장하고 불러오는 경우:

    ```python
    # 저장
    torch.save(model.state_dict(), 'model_weights.pt')
    
    # 불러오기
    model.load_state_dict(torch.load('model_weights.pt'))
    ```

  - 전체 모델을 저장하고 불러오는 경우:

    ```python
    # 저장
    torch.save(model, 'model.pt')
    
    # 불러오기
    model = torch.load('model.pt')
    ```

    
