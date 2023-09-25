import torch
import torch.nn as nn

# GPU 사용 가능한지 확인
if torch.cuda.is_available():
    # GPU를 사용할 디바이스로 설정
    device = torch.device("cuda")
    print("GPU를 사용합니다.")
else:
    # GPU를 사용할 수 없으면 CPU를 사용
    device = torch.device("cpu")
    print("CPU를 사용합니다.")

# 예제 텐서를 GPU 또는 CPU로 옮기기
x = torch.randn(3, 3).to(device)

# 모델 정의: 간단한 선형 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 2)  # 3차원 입력을 2차원으로 변환

    def forward(self, x):
        return self.fc(x)

# 모델을 GPU 또는 CPU로 옮기기
model = SimpleModel().to(device)

# 모델 실행
output = model(x)
print(output)