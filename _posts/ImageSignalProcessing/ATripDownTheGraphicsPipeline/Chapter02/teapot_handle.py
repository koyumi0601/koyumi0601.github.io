import numpy as np
import matplotlib.pyplot as plt

# 데이터 포인트 정의 - 손잡이 (4개씩)
handle_points = {
    "curve1": {
        "x": np.array([-1.60, -2.30, -2.70, -2.70]),
        "y": np.array([1.8750, 1.8750, 1.8750, 1.6500])
    },
    "curve2": {
        "x": np.array([-2.70, -2.70, -2.50, -2.00]),
        "y": np.array([1.6500, 1.4250, 0.9750, 0.7500])
    },
    "curve3": {
        "x": np.array([-1.50, -2.50, -3.00, -3.00]),
        "y": np.array([2.1000, 2.1000, 2.1000, 1.6500])
    },
    "curve4": {
        "x": np.array([-3.00, -3.00, -2.65, -1.90]),
        "y": np.array([1.6500, 1.2000, 0.7875, 0.4500])
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

for curve_name, points in handle_points.items():
    x = points["x"]
    y = points["y"]
    
    curve_x = []
    curve_y = []
    
    for t in t_values:
        curve_x.append(bezier_curve(t, x))
        curve_y.append(bezier_curve(t, y))
    
    plt.plot(curve_x, curve_y, label=f'bezier {curve_name}')

# 데이터 포인트를 직선으로 잇는 그래프 추가
for curve_name, points in handle_points.items():
    x = points["x"]
    y = points["y"]
    
    plt.plot(x, y, 'o-', label=f'data point {curve_name}')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Figure 2.3 Outline of the handle')
plt.legend()
plt.grid()
plt.show()