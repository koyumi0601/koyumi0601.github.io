# import numpy as np
# from scipy.interpolate import griddata

# # 예제 데이터 생성
# x = np.random.rand(3500000)  # 임의의 x 좌표
# y = np.random.rand(3500000)  # 임의의 y 좌표
# z = np.random.rand(3500000)  # 임의의 z 좌표
# v = np.random.rand(3500000)  # 임의의 값

# print(f'x: {x[0:10]}')
# print(f'y: {y[0:10]}')
# print(f'z: {z[0:10]}')
# print(f'v: {v[0:10]}')

# # x', y', z' 좌표 정의 (여기서는 임의의 좌표를 사용)
# x_new = np.random.rand(10000)  # 임의의 x' 좌표
# y_new = np.random.rand(10000)  # 임의의 y' 좌표
# z_new = np.random.rand(10000)  # 임의의 z' 좌표



# # 보간 수행
# v_new = griddata((x, y, z), v, (x_new, y_new, z_new), method='linear', fill_value=0)

# print(f'x_new: {x_new[0:10]}')
# print(f'y_new: {y_new[0:10]}')
# print(f'z_new: {z_new[0:10]}')
# print(f'v_new: {v_new[0:10]}')



# # v_new에는 x', y', z' 좌표에 대응하는 값을 가지고 있음


import numpy as np
from scipy.interpolate import interpn
def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z
x = np.linspace(0, 4, 650)
y = np.linspace(0, 5, 650)
z = np.linspace(0, 6, 400)
points = (x, y, z)
values = value_func_3d(*np.meshgrid(*points, indexing='ij'))

# point = np.array([2.21, 3.12, 1.15])
point_x = np.array(np.random.rand(10000))
point_y = np.array(np.random.rand(10000))
point_z = np.array(np.random.rand(10000))
point = np.array([point_x, point_y, point_z]).T
# print(point.shape)
print(interpn(points, values, point))