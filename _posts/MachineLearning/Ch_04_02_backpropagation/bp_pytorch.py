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