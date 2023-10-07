import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 데이터 전처리
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 로컬에서 데이터셋 불러오기
with open('trainset.pkl', 'rb') as f:
    trainset = pickle.load(f)
with open('testset.pkl', 'rb') as f:
    testset = pickle.load(f)


# 강아지와 고양이만 선택
indices = [i for i, target in enumerate(trainset.targets) if target == 3 or target == 5]
trainset.data = [trainset.data[i] for i in indices]
trainset.targets = [0 if trainset.targets[i] == 3 else 1 for i in indices]  # 강아지는 0, 고양이는 1로 매핑

indices = [i for i, target in enumerate(testset.targets) if target == 3 or target == 5]
testset.data = [testset.data[i] for i in indices]
testset.targets = [0 if testset.targets[i] == 3 else 1 for i in indices]  # 강아지는 0, 고양이는 1로 매핑

print(f"Number of training images: {len(trainset)}")
print(f"Number of testing images: {len(testset)}")


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

# # 이미지를 플롯하기 위한 함수
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # 학습 데이터의 일부를 가져옵니다.
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # 이미지를 표시합니다.
# imshow(images[3])


# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 2)  # RGB 이미지, 32x32 크기

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc(x)
        return x

net = Net()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.2)

# 학습
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print('Finished Training')

# 평가
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test images: {100 * correct / total}%')