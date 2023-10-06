import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 정의
def unit_step(x):
    return np.where(x > 0, 1, 0)

def linear_activation(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 데이터 생성
x = np.linspace(-10, 10, 400)

# 활성화 함수 플롯
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, unit_step(x))
plt.title('Unit Step Function')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, linear_activation(x))
plt.title('Linear Activation')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, tanh(x))
plt.title('Hyperbolic Tangent (tanh)')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, relu(x))
plt.title('Rectified Linear Unit (ReLU)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.tight_layout()
plt.show()