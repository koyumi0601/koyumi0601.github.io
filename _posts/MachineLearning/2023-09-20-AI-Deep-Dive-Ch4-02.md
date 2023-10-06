---
layout: single
title: "AI Deep Dive, Chapter 4. 딥러닝, 그것이 알고싶다 02. Backpropagation 깊은 인공신경망의 학습"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 4 - 02. Backpropagation 깊은 인공신경망의 학습



gradient descent를 하기 위해서, **미분**은 어떻게 구할까?

- 모델에서, 모두 편미분을 해서, 미분을 구해야할 것
- 몇 번이나? w, b 총 17개



![ch0402_1]({{site.url}}/images/$(filename)/ch0402_1.png)



- 17x1 짜리 gradient 를 구해야한다
  - learning rate만큼 곱해서, 업데이트를 해서, 가주면 됨
- 이런 편미분을 할 때, **backpropagation**이라는 것을 해야한다.
  - chain rule인데, 방향이 뒤에서부터 앞으로가서, backpropagation이라고 부르는 것

![ch0402_2]({{site.url}}/images/$(filename)/ch0402_2.png)





- 맨 바깥 출력층부터 한번 표현해보자면(빨간색)
  - $$ d_2 $$: 들어가는 것
  - $$ n_2 $$: 나가는 것
  - $$ w_1 $$: weight, 얘에 대한 Loss의 편미분을 구해보자
  - $$ b_2 $$: bias
  - $$ f_2 $$: activation function
  - $$ \hat{y_1} $$: 최종 출력, estimated
  - $$ y_1 $$: 참값
  - $$ \hat{y_2} $$: 최종 출력, estimated
  - $$ y_2 $$: 참값
  - Loss: 일단 MSE로 가정, 참값과 estimation 값의 차이의 제곱

![ch0402_7]({{site.url}}/images/$(filename)/ch0402_7.png)

![ch0402_9]({{site.url}}/images/$(filename)/ch0402_9.png)



- 이번엔, $$ w_2 $$에 대한 편미분을 구해보자

  ![ch0402_12]({{site.url}}/images/$(filename)/ch0402_12.png)

  - path 1

    ![ch0402_10]({{site.url}}/images/$(filename)/ch0402_10.png)

    ![ch0402_11]({{site.url}}/images/$(filename)/ch0402_11.png)

    ![ch0402_13]({{site.url}}/images/$(filename)/ch0402_13.png)

    출력 * 액(티베이션의 미분) * 웨(이트) * 액(티베이션의 미분) * 앤(입력)

  - path 2

    







![ch0402_4]({{site.url}}/images/$(filename)/ch0402_4.png)



![ch0402_5]({{site.url}}/images/$(filename)/ch0402_5.png)





정리하자면, forward propagation을 한번 해서, 값들을 구해놓고, backward propagation을 통해서 미분을 구해야 한다.

무슨 말?

- $$ d_1, d_2 $$를 구하려면 데이터를 넣어봐야 한다.

![ch0402_6]({{site.url}}/images/$(filename)/ch0402_6.png)





# 추가 예제

```python
import torch
import time
from memory_profiler import profile

# 시작 시간 기록
start_time = time.time()

@profile
def main():
    # 데이터셋 생성
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    # 모델 파라미터 초기화
    w = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)

    # 학습률 설정
    learning_rate = 0.01

    # 학습 루프
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 순전파 계산
        predictions = x.mm(w) + b
        
        # 손실 계산
        loss = ((predictions - y) ** 2).mean()

        # 그래디언트 계산, backpropagation
        dw = 2 * x.t().mm(predictions - y) / x.size(0)
        db = 2 * (predictions - y).sum() / x.size(0)

        # 가중치 업데이트
        with torch.no_grad():
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # 로그 출력
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 최종 학습된 모델의 가중치 출력
    print("최종 학습된 가중치 w:", w.item())
    print("최종 학습된 편향 b:", b.item())

if __name__ == "__main__":
    main()

# 종료 시간 기록
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time

print(f"코드 실행 시간: {execution_time:.4f} 초")
```



##### 출력

```python
Epoch [100/1000], Loss: 0.0076
Epoch [200/1000], Loss: 0.0042
Epoch [300/1000], Loss: 0.0023
Epoch [400/1000], Loss: 0.0013
Epoch [500/1000], Loss: 0.0007
Epoch [600/1000], Loss: 0.0004
Epoch [700/1000], Loss: 0.0002
Epoch [800/1000], Loss: 0.0001
Epoch [900/1000], Loss: 0.0001
Epoch [1000/1000], Loss: 0.0000
최종 학습된 가중치 w: 2.0048611164093018
최종 학습된 편향 b: 0.9857079386711121
Filename: bp.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     8    290.3 MiB    290.3 MiB           1   @profile
     9                                         def main():
    10                                             # 데이터셋 생성
    11    290.3 MiB      0.0 MiB           1       x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    12    290.3 MiB      0.0 MiB           1       y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)
    13                                         
    14                                             # 모델 파라미터 초기화
    15    290.3 MiB      0.0 MiB           1       w = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    16    290.3 MiB      0.0 MiB           1       b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    17                                         
    18                                             # 학습률 설정
    19    290.3 MiB      0.0 MiB           1       learning_rate = 0.01
    20                                         
    21                                             # 학습 루프
    22    290.3 MiB      0.0 MiB           1       num_epochs = 1000
    23    297.6 MiB      0.0 MiB        1001       for epoch in range(num_epochs):
    24                                                 # 순전파 계산
    25    297.6 MiB      1.5 MiB        1000           predictions = x.mm(w) + b
    26                                                 
    27                                                 # 손실 계산
    28    297.6 MiB      2.8 MiB        1000           loss = ((predictions - y) ** 2).mean()
    29                                         
    30                                                 # 그래디언트 계산, backpropagation
    31    297.6 MiB      0.0 MiB        1000           dw = 2 * x.t().mm(predictions - y) / x.size(0)
    32    297.6 MiB      0.0 MiB        1000           db = 2 * (predictions - y).sum() / x.size(0)
    33                                         
    34                                                 # 가중치 업데이트
    35    297.6 MiB      0.0 MiB        1000           with torch.no_grad():
    36    297.6 MiB      0.0 MiB        1000               w -= learning_rate * dw
    37    297.6 MiB      0.0 MiB        1000               b -= learning_rate * db
    38                                                 
    39                                                 # 로그 출력
    40    297.6 MiB      0.0 MiB        1000           if (epoch + 1) % 100 == 0:
    41    297.6 MiB      3.0 MiB          10               print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    42                                         
    43                                             # 최종 학습된 모델의 가중치 출력
    44    297.6 MiB      0.0 MiB           1       print("최종 학습된 가중치 w:", w.item())
    45    297.6 MiB      0.0 MiB           1       print("최종 학습된 편향 b:", b.item())


코드 실행 시간: 0.8926 초
```



##### pytorch

```python
import torch
import time
from memory_profiler import profile

# 시작 시간 기록
start_time = time.time()

@profile
def main():
    # 데이터셋 생성
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    # 모델 파라미터 초기화
    w = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)

    # 학습률 설정
    learning_rate = 0.01

    # 손실 함수 정의 (MSE Loss)
    loss_fn = torch.nn.MSELoss()

    # 옵티마이저 정의 (확률적 경사 하강법 SGD)
    optimizer = torch.optim.SGD([w, b], lr=learning_rate)

    # 학습 루프
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 순전파 계산
        predictions = x.mm(w) + b

        # 손실 계산
        loss = loss_fn(predictions, y)

        # 그래디언트 계산 및 역전파
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트

        # 로그 출력
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 최종 학습된 모델의 가중치 출력
    print("최종 학습된 가중치 w:", w.item())
    print("최종 학습된 편향 b:", b.item())

if __name__ == "__main__":
    main()

# 종료 시간 기록
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time

print(f"코드 실행 시간: {execution_time:.4f} 초")
```



##### 실행결과, performance check

```python
Filename: bp_pytorch.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     8    292.9 MiB    292.9 MiB           1   @profile
     9                                         def main():
    10                                             # 데이터셋 생성
    11    292.9 MiB      0.0 MiB           1       x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    12    292.9 MiB      0.0 MiB           1       y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)
    13                                         
    14                                             # 모델 파라미터 초기화
    15    292.9 MiB      0.0 MiB           1       w = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    16    292.9 MiB      0.0 MiB           1       b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32)
    17                                         
    18                                             # 학습률 설정
    19    292.9 MiB      0.0 MiB           1       learning_rate = 0.01
    20                                         
    21                                             # 손실 함수 정의 (MSE Loss)
    22    292.9 MiB      0.0 MiB           1       loss_fn = torch.nn.MSELoss()
    23                                         
    24                                             # 옵티마이저 정의 (확률적 경사 하강법 SGD)
    25    292.9 MiB      0.0 MiB           1       optimizer = torch.optim.SGD([w, b], lr=learning_rate)
    26                                         
    27                                             # 학습 루프
    28    292.9 MiB      0.0 MiB           1       num_epochs = 1000
    29    301.9 MiB      0.0 MiB        1001       for epoch in range(num_epochs):
    30                                                 # 순전파 계산
    31    301.9 MiB      1.3 MiB        1000           predictions = x.mm(w) + b
    32                                         
    33                                                 # 손실 계산
    34    301.9 MiB      2.8 MiB        1000           loss = loss_fn(predictions, y)
    35                                         
    36                                                 # 그래디언트 계산 및 역전파
    37    301.9 MiB      0.0 MiB        1000           optimizer.zero_grad()  # 그래디언트 초기화
    38    301.9 MiB      5.0 MiB        1000           loss.backward()  # 역전파
    39    301.9 MiB      0.0 MiB        1000           optimizer.step()  # 가중치 업데이트
    40                                         
    41                                                 # 로그 출력
    42    301.9 MiB      0.0 MiB        1000           if (epoch + 1) % 100 == 0:
    43    301.9 MiB      0.0 MiB          10               print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    44                                         
    45                                             # 최종 학습된 모델의 가중치 출력
    46    301.9 MiB      0.0 MiB           1       print("최종 학습된 가중치 w:", w.item())
    47    301.9 MiB      0.0 MiB           1       print("최종 학습된 편향 b:", b.item())


코드 실행 시간: 1.1261 초
```



