import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.array([[160, 45], [162, 51], [165, 55], [170, 63], [180, 75]], dtype=np.float32)
x_train = data[:, 0]  # 키
y_train = data[:, 1]  # 몸무게

# 주어진 범위 내에서 a와 b의 조합을 시도하며 MSE 값을 저장
a_range = np.arange(0, 10, 0.1)
b_range = np.arange(-1000, 100, 1)
mse_values = np.zeros((len(a_range), len(b_range)))

for i, a in enumerate(a_range):
    for j, b in enumerate(b_range):
        # 현재 a와 b를 사용하여 예측값 계산
        y_pred = a * x_train + b

        # MSE 계산
        mse = np.mean((y_pred - y_train) ** 2)
        mse_values[i, j] = mse

# a와 b 값에 대한 그리드 생성
a_grid, b_grid = np.meshgrid(a_range, b_range)

# 최소 MSE 값을 찾아 해당 위치 저장
min_mse = np.min(mse_values)
min_mse_idx = np.unravel_index(np.argmin(mse_values, axis=None), mse_values.shape)
min_a = a_range[min_mse_idx[0]]
min_b = b_range[min_mse_idx[1]]

# 3D surface plot으로 MSE 값 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(a_grid, b_grid, mse_values.T, cmap='viridis')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('MSE Surface Plot')

# 최소 MSE 지점을 점으로 표시
ax.scatter(min_a, min_b, min_mse, color='red', marker='o', s=100, label='Min MSE')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()
plt.show()

# 최소 MSE를 갖는 a와 b 값 출력
print(f"Min MSE: {min_mse}")
print(f"Min a: {min_a}")
print(f"Min b: {min_b}")