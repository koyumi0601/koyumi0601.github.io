import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.linspace(0, 200, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
Z = np.where(Y > X + 1, 1, 0)

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Need to Lose Weight')
ax.set_title('3D Plot of Weight vs Height')

# 각 축의 범위 설정
ax.set_xlim([0, 200])
ax.set_ylim([0, 100])
ax.set_zlim([-1, 2])

plt.show()