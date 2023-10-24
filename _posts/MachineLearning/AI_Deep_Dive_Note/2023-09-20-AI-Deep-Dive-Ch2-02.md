---
layout: single
title: "AI Deep Dive, Chapter 2. 왜 현재 AI가 가장 핫할까? 02. 딥러닝의 활용 CNN, RNN, GAN"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 2 - 02. 딥러닝의 활용 CNN, RNN, GAN



## CNN (Convolutional Neural Network)

![CNN]({{site.url}}/images/$(filename)/CNN.png)


- label: dog 1, cat 0

- input image(Gray scale): matrix, number

    ![input_image_gray_scale]({{site.url}}/images/$(filename)/input_image_gray_scale.png)

- input image(RGB): matrix, 3D, number
    
    ![input_image_color]({{site.url}}/images/$(filename)/input_image_color.png)
    
    - 규칙: channel 수가 가장 앞이고, 그 다음이 이미지 사이즈이다. 3x5x5. 채 행 열
    - 이미지 두 장이면, 2x3x5x5 개 채 행 열
    - 흑백 사진이면, 채널 개수가 1임.



#### 구현 예시: MNIST 데이터셋에 대한 손글씨 숫자 분류

1. MNIST 데이터셋을 로드, 전처리
2. CNN 모델 정의
   1. 3개의 Conv2D 레이어, 2개의 MaxPooling2D 레이어
3. 모델을 컴파일, 학습 데이터로 학습
4. 테스트 데이터로 모델의 성능 평가
5. 모델의 테스트 정확도 출력

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 데이터 로딩 및 전처리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# CNN 모델 구성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_images, train_labels, epochs=10, batch_size=64)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc * 100:.2f}%')
```





## RNN (Recurrent Neural network)

연속적인 데이터에 대해서 잘 동작
문장

![RNN]({{site.url}}/images/$(filename)/RNN.png)

입력, 출력이 문장이니 숫자가 아닐까? 

- 저는 강사 입니다. -> 글자를 숫자로 바꿔준다. 100, 010, 001이라고 바꾼다고 하면, 연속된 숫자가 들어간다고 할 수 있다.
- I am an instructor -> 1000 0100 0010 0001 이라고 바꾼다고 할 수 있다. 실제로는 영어라고 하면 5000몇개 된다.






## GAN (Generative Adversarial Network)

![GAN]({{site.url}}/images/$(filename)/GAN.png)



G, D라는 작은 두 개의 네트워크로 구성되어 있다.

G: 위조지폐를 만드는 네트워크. 

D: 위조지폐인지, 진짜 지폐인지 가려냄. 학습 시킴. 기존 네트워크를 서브로 갖다 써도 됨. CNN을 갖다 쓰면 됨.

학습의 방향이 서로 다르다.  - 적대적 신경망



목표: D는 위조지폐이면 0을 뱉도록 학습해야 하고, G는 D속게끔 학습이 되어야 한다.



둘은 동시에 학습할 수 없다.

한쪽이 학습하는 동안, 다른 쪽은 중단해야 한다.



D부터 학습한다고 보면,

G는 가만히 있는다. 이상한 이미지를 생성한다.

D는 이상한 이미지와, 진짜 이미지를 구별하는 문제를 푼다. 쉬울 것. D의 실력이 조금 올라간다.

D는 가만히 있는다. G를 학습시킨다. 학습 초기니까 D를 속이는 것은 쉬울 것. 실력이 조금 올라간다. D의 출력이 1(진짜 지폐)이 나오도록 학습한다.



학습이 완료되면, D가 진짜도 0.5, 가짜도 0.5 = 모르겠다고 하는 것이 수렴의 결과이다.



![GAN_result]({{site.url}}/images/$(filename)/GAN_result.png)



원하는 것은, D가 학습한 결과보다는 G가 만든 리얼한 가짜 이미지이다.



![GAN_want]({{site.url}}/images/$(filename)/GAN_want.png)



##### GAN 예시



![GAN_youtube]({{site.url}}/images/$(filename)/GAN_youtube.png)

스케치를 넣으면, 진짜 사람이 나옴 Edge to Face

- ground truth



Wang et al. 2018 - Video to video Synthesis

![GAN_wang]({{site.url}}/images/$(filename)/GAN_wang.png)



Pose to Body

![GAN_pose_to_body]({{site.url}}/images/$(filename)/GAN_pose_to_body.png)