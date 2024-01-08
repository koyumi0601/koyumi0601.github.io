from scipy.interpolate import RegularGridInterpolator
import numpy as np
def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
# xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij')
# xg, yg = np.meshgrid(x, y, indexing='ij')
data = f(xg, yg, zg)
interp = RegularGridInterpolator((x, y, z), data)


pts = np.array([[2.1, 6.2, 8.3],
                [3.3, 5.2, 7.1]])


# print(f' coordinates: {x, y, z}')
# print(f' intp: {interp(pts)}') 
# print(f' ground truth: {f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)}')
print(f'xg.shape: {xg.shape}, yg.shape: {yg.shape}, zg.shape={zg.shape}')
x2 = np.linspace(1, 1.5, 11)
y2 = np.linspace(4, 4.5, 22)
z2 = np.linspace(7, 7.3, 33)
xg2, yg2 ,zg2 = np.meshgrid(x2, y2, z2, indexing='ij')
print(f'xg2.shape: {xg2.shape}, yg2.shape: {yg2.shape}, zg2.shape={zg.shape}')
print(xg2)
# array([ 125.80469388,  146.30069388])
pts2 = (xg2, yg2, zg2)
output = interp(pts2)

# print(output[1,1,1])
# print(xg2.shape[2])
# print(range(0, xg2.shape[2]))
ids = np.linspace(0, xg2.shape[2]-1, xg2.shape[2])
# print(ids)