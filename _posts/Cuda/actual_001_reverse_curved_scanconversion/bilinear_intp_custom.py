import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(4)
xg, yg = np.meshgrid(x, y)
v = xg**2 + yg**2

# plt.imshow(v, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.imshow(v)
plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('2D Grid Data')
plt.show()