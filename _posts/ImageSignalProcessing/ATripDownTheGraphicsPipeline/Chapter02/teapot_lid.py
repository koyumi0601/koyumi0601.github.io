import numpy as np
import matplotlib.pyplot as plt

# 데이터 포인트 정의 - 첫 번째 곡선
curve1_points = {
    "x": np.array([0.0, 0.8, 0.0, 0.2]),
    "y": np.array([3.00, 3.00, 2.70, 2.55])
}

# 데이터 포인트 정의 - 두 번째 곡선
curve2_points = {
    "x": np.array([0.2, 0.4, 1.3, 1.3]),
    "y": np.array([2.55, 2.40, 2.40, 2.25])
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

# 첫 번째 곡선 그리기
x1 = curve1_points["x"]
y1 = curve1_points["y"]

curve_x1 = []
curve_y1 = []

for t in t_values:
    curve_x1.append(bezier_curve(t, x1))
    curve_y1.append(bezier_curve(t, y1))

plt.plot(curve_x1, curve_y1, label='bezier curve 1')

# 두 번째 곡선 그리기
x2 = curve2_points["x"]
y2 = curve2_points["y"]

curve_x2 = []
curve_y2 = []

for t in t_values:
    curve_x2.append(bezier_curve(t, x2))
    curve_y2.append(bezier_curve(t, y2))

plt.plot(curve_x2, curve_y2, label='bezier curve 2')

# 데이터 포인트를 직선으로 연결하는 그래프 추가
plt.plot(x1, y1, 'ro-', label='data point 1 (linear)')
plt.plot(x2, y2, 'go-', label='data point 2 (linear)')

plt.xlabel('X axis')
plt.ylabel('Y aixs')
plt.title('Figure 2.2 Outline of the lid')
plt.legend()
plt.grid()
plt.show()