from scipy.interpolate import RegularGridInterpolator
import numpy as np

x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
def ff(x, y):
    return x**2 + y**2

xg, yg = np.meshgrid(x, y, indexing='ij')
data = ff(xg, yg) 
interp = RegularGridInterpolator((x, y), data,
                                 bounds_error=False, fill_value=None)
# Regular grid에 대해서, x축, y축, x-y grid에 대한 데이터를 넣어서 intp 인스턴스 생성
# 옵션은 boundary를 넘어가는 경우 값을 채울 것인가(bounds_error)? 채운다면 무엇으로(fill_value)?
print(f'xg.shape: {xg.shape}, yg.shape: {yg.shape}')
##

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
           s=60, c='k', label='data')
xx = np.linspace(-4, 9, 31)
yy = np.linspace(-4, 9, 31)
X, Y = np.meshgrid(xx, yy, indexing='ij')

print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')


# interpolator
ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
# ground truth
ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
                  alpha=0.4, label='ground truth')
plt.legend()
plt.show()