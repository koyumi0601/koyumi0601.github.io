import tensorflow as tf  # TensorFlow 라이브러리를 불러옵니다.
from tensorflow.keras import layers, models  # Keras에서 필요한 layers와 models를 불러옵니다.
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋을 불러옵니다.
from tensorflow.keras.utils import to_categorical  # 범주형 데이터를 다루기 위한 유틸리티를 불러옵니다.

# # 데이터 로딩 및 전처리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # MNIST 데이터를 로드합니다.
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255  # 이미지 데이터를 전처리합니다.
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255  # 테스트 이미지 데이터를 전처리합니다.

train_labels = to_categorical(train_labels)  # 레이블을 범주형으로 변환합니다.
test_labels = to_categorical(test_labels)  # 테스트 레이블을 범주형으로 변환합니다.

# CNN 모델 구성
model = models.Sequential()  # Sequential 모델을 생성합니다.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 첫 번째 Conv2D 레이어를 추가합니다.
model.add(layers.MaxPooling2D((2, 2)))  # MaxPooling 레이어를 추가합니다.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 두 번째 Conv2D 레이어를 추가합니다.
model.add(layers.MaxPooling2D((2, 2)))  # 두 번째 MaxPooling 레이어를 추가합니다.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 세 번째 Conv2D 레이어를 추가합니다.
model.add(layers.Flatten())  # Flatten 레이어를 추가하여 3D 출력을 1D로 변환합니다.
model.add(layers.Dense(64, activation='relu'))  # Dense 레이어를 추가합니다.
model.add(layers.Dense(10, activation='softmax'))  # 출력 레이어를 추가합니다.

# 모델 컴파일
model.compile(optimizer='adam',  # 옵티마이저로 Adam을 사용합니다.
              loss='categorical_crossentropy',  # 손실 함수로 categorical_crossentropy를 사용합니다.
              metrics=['accuracy'])  # 평가 지표로 정확도를 사용합니다.

# 모델 학습
model.fit(train_images, train_labels, epochs=10, batch_size=64)  # 모델을 학습시킵니다. Need cuDNN libraries

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)  # 테스트 데이터로 모델을 평가합니다.
print(f'Test accuracy: {test_acc * 100:.2f}%')  # 테스트 정확도를 출력합니다.