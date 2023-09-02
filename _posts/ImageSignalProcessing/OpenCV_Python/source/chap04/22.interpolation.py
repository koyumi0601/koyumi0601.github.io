import matplotlib.pyplot as plt
import numpy as np

methods = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36' ]
grid = np.random.rand(5, 5)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))

for ax, method in zip(axs.flat, methods):
    ax.imshow(grid, interpolation=method, cmap='gray')
    ax.set_title(method)
plt.tight_layout(), plt.show()