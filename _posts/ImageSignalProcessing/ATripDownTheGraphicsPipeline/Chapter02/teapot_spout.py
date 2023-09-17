import numpy as np
import matplotlib.pyplot as plt

# 데이터 포인트 정의 - 스포트
spout_points = {
    "curve1": {
        "x": np.array([1.700, 2.600, 2.300, 2.700]),
        "y": np.array([1.27500, 1.27500, 1.95000, 2.25000])
    },
    "curve2": {
        "x": np.array([2.700, 2.800, 2.900, 2.800]),
        "y": np.array([2.25000, 2.32500, 2.32500, 2.25000])
    },
    "curve3": {
        "x": np.array([1.700, 3.100, 2.400, 3.300]),
        "y": np.array([0.45000, 0.67500, 1.87500, 2.25000])
    },
    "curve4": {
        "x": np.array([3.300, 3.525, 3.450, 3.200]),
        "y": np.array([2.25000, 2.34375, 2.36250, 2.25000])
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

for curve_name, points in spout_points.items():
    x = points["x"]
    y = points["y"]
    
    curve_x = []
    curve_y = []
    
    for t in t_values:
        curve_x.append(bezier_curve(t, x))
        curve_y.append(bezier_curve(t, y))
    
    plt.plot(curve_x, curve_y, label=f'Bezier {curve_name}')

# 데이터 포인트를 직선으로 잇는 그래프 추가
for curve_name, points in spout_points.items():
    x = points["x"]
    y = points["y"]
    
    plt.plot(x, y, 'o-', label=f'Data Points {curve_name}')

# 직선 그래프 그리기
for curve_name, points in spout_points.items():
    x = points["x"]
    y = points["y"]
    
    # plt.plot(x, y, '--', label=f'Straight Line {curve_name}')

# 타이틀, 축 이름, 레이블 영어로 설정
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Figure 2.4 Outline of the spout')
plt.legend()
plt.grid()
plt.show()