import torch
import time
from memory_profiler import profile

# 시작 시간 기록
start_time = time.time()

def main():
    # 데이터셋 생성 및 GPU로 이동
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32).cuda()
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32).cuda()

    # 모델 파라미터 초기화 및 GPU로 이동
    w = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32).cuda()
    b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float32).cuda()

    # 학습률 설정
    learning_rate = 0.01

    # 손실 함수 정의 (MSE Loss)
    loss_fn = torch.nn.MSELoss()

    # 옵티마이저 정의 (확률적 경사 하강법 SGD)
    optimizer = torch.optim.SGD([w.detach(), b.detach()], lr=learning_rate)  # w와 b를 leaf Tensor로 변환

    # 학습 루프
    num_epochs = 1000

    current_memory = torch.cuda.memory_allocated()
    print(f"현재 할당된 GPU 메모리(for문 전): {current_memory / 1024**2} MB")

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

        # 메모리 사용량 측정 (GPU)
    current_memory = torch.cuda.memory_allocated()
    print(f"현재 할당된 GPU 메모리: {current_memory / 1024**2} MB")

    # 최종 학습된 모델의 가중치 출력
    print("최종 학습된 가중치 w:", w.item())
    print("최종 학습된 편향 b:", b.item())

if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        # CUDA 장치 설정
        device = torch.device("cuda")
        print("GPU 사용 가능")
    else:
        device = torch.device("cpu")
        print("GPU 사용 불가능")

    # 코드 내 모든 Tensor와 모델을 GPU로 이동
    main()

    # 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산
    execution_time = end_time - start_time

    print(f"코드 실행 시간: {execution_time:.4f} 초")
    torch.cuda.empty_cache()
    current_memory = torch.cuda.memory_allocated()
    print(f"현재 할당된 GPU 메모리, 실행후: {current_memory / 1024**2} MB")