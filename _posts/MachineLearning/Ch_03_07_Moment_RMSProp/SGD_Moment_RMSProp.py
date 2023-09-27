import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# 초기값 및 하이퍼파라미터 설정
w1_init = 1.0
w2_init = 1.0
learning_rate = 0.1
num_steps = 50

# Optimizer 클래스 정의
class SGDOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, w1, w2, grad_w1, grad_w2):
        w1 -= self.learning_rate * grad_w1
        w2 -= self.learning_rate * grad_w2
        return w1, w2

class MomentumOptimizer:
    def __init__(self, learning_rate, alpha=0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.v_w1 = 0
        self.v_w2 = 0
    
    def update(self, w1, w2, grad_w1, grad_w2):
        self.v_w1 = self.alpha * self.v_w1 + self.learning_rate * grad_w1
        self.v_w2 = self.alpha * self.v_w2 + self.learning_rate * grad_w2
        w1 -= self.v_w1
        w2 -= self.v_w2
        return w1, w2

class RMSPropOptimizer:
    def __init__(self, learning_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.s_w1 = 0
        self.s_w2 = 0
    
    def update(self, w1, w2, grad_w1, grad_w2):
        self.s_w1 = self.beta * self.s_w1 + (1 - self.beta) * grad_w1 ** 2
        self.s_w2 = self.beta * self.s_w2 + (1 - self.beta) * grad_w2 ** 2
        w1 -= self.learning_rate * grad_w1 / (np.sqrt(self.s_w1) + 1e-10)
        w2 -= self.learning_rate * grad_w2 / (np.sqrt(self.s_w2) + 1e-10)
        return w1, w2

# 손실 함수와 그래디언트 정의
def f(w1, w2):
    return w1**2 + w2**2

def gradient_w1(w1):
    return 2 *  w1

def gradient_w2(w2):
    return 2 * w2

# Optimizer 인스턴스 생성
sgd_optimizer = SGDOptimizer(learning_rate)
momentum_optimizer = MomentumOptimizer(learning_rate)
rmsprop_optimizer = RMSPropOptimizer(learning_rate)

# 경로 저장을 위한 리스트 초기화
sgd_path = [(w1_init, w2_init)]
momentum_path = [(w1_init, w2_init)]
rmsprop_path = [(w1_init, w2_init)]

# 경사 하강법 수행
w1_sgd, w2_sgd = w1_init, w2_init
w1_momentum, w2_momentum = w1_init, w2_init
w1_rmsprop, w2_rmsprop = w1_init, w2_init

for step in range(num_steps):

    grad_sgd_w1 = gradient_w1(w1_sgd)
    grad_sgd_w2 = gradient_w2(w2_sgd)

    grad_momentum_w1 = gradient_w1(w1_momentum)
    grad_momentum_w2 = gradient_w2(w2_momentum)

    grad_rmsprop_w1 = gradient_w1(w1_rmsprop)
    grad_rmsprop_w2 = gradient_w2(w2_rmsprop)

    w1_sgd, w2_sgd = sgd_optimizer.update(w1_sgd, w2_sgd, grad_sgd_w1, grad_sgd_w2)
    sgd_path.append((w1_sgd, w2_sgd))
    
    w1_momentum, w2_momentum = momentum_optimizer.update(w1_momentum, w2_momentum, grad_momentum_w1, grad_momentum_w2)
    momentum_path.append((w1_momentum, w2_momentum))
    
    w1_rmsprop, w2_rmsprop = rmsprop_optimizer.update(w1_rmsprop, w2_rmsprop, grad_rmsprop_w1, grad_rmsprop_w2)
    rmsprop_path.append((w1_rmsprop, w2_rmsprop))

# 그래프 생성 및 플로팅
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(15, 5))

def plot_path(ax, path, title, color):
    ax.set_title(title)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    x, y = zip(*path)
    num_points = len(path)
    colors = [mcolors.to_rgba(color, alpha=i/num_points) for i in range(num_points)]  # 그라데이션 적용
    ax.scatter(x, y, [f(w1, w2) for w1, w2 in path], c=colors, marker='o', s=30)
    ax.scatter(x[-1], y[-1], [f(w1, w2) for w1, w2 in [path[-1]]], c='red', marker='o', s=50)  # 최종 도착 지점을 다른 색으로 표시
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Loss Function')

ax1 = fig.add_subplot(131, projection='3d')
plot_path(ax1, sgd_path, 'SGD for $f(w1, w2) = w1^2 + w2^2$', 'blue')

ax2 = fig.add_subplot(132, projection='3d')
plot_path(ax2, momentum_path, 'Momentum for $f(w1, w2) = w1^2 + w2^2$', 'blue')

ax3 = fig.add_subplot(133, projection='3d')
plot_path(ax3, rmsprop_path, 'RMSProp for $f(w1, w2) = w1^2 + w2^2$', 'blue')

plt.show()