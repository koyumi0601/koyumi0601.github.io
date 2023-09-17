import numpy as np
import matplotlib.pyplot as plt

# 데이터 포인트 정의
data_points = {
    "curve1": {
        "x": np.array([1.4000, 1.3375, 1.4375, 1.5000]),
        "y": np.array([2.25000, 2.38125, 2.38125, 2.25000])
    },
    "curve2": {
        "x": np.array([1.5000, 1.7500, 2.0000, 2.0000]),
        "y": np.array([2.25000, 1.72500, 1.20000, 0.75000])
    },
    "curve3": {
        "x": np.array([2.0000, 2.0000, 1.5000, 1.5000]),
        "y": np.array([0.75000, 0.30000, 0.07500, 0.00000])
    }
}

# 베지어 곡선 함수 정의
def bezier_curve(t, control_points):
    n = len(control_points) - 1
    result = 0
    for i, p in enumerate(control_points):
        result += p * (np.math.comb(n, i)) * ((1 - t) ** (n - i)) * (t ** i)
    return result

# 베지어 곡선 그리기
t_values = np.linspace(0, 1, 1000)

plt.figure()

for curve_name, points in data_points.items():
    x = points["x"]
    y = points["y"]
    
    curve_x = []
    curve_y = []
    
    for t in t_values:
        curve_x.append(bezier_curve(t, x))
        curve_y.append(bezier_curve(t, y))
    
    plt.plot(curve_x, curve_y, label=f'bezier {curve_name}')

plt.scatter([1.4000, 1.3375, 1.4375, 1.5000, 1.7500, 2.0000, 2.0000, 2.0000, 1.5000, 1.5000],
            [2.25000, 2.38125, 2.38125, 2.25000, 1.72500, 1.20000, 0.75000, 0.30000, 0.07500, 0.00000],
            label='Control Point', color='black')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Outline of the body')
plt.legend()
plt.grid()
plt.show()