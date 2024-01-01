import numpy as np

def cubic_spline(x):
    # 삼차 스플라인 다항식 계산
    if abs(x) <= 1:
        return (1.5 * abs(x)**3 - 2.5 * abs(x)**2 + 1)
    elif 1 < abs(x) <= 2:
        return (-0.5 * abs(x)**3 + 2.5 * abs(x)**2 - 4 * abs(x) + 2)
    else:
        return 0

def tricubic_spline_interpolation(volume_data, x, y, z):
    x = max(0, min(volume_data.shape[0] - 1, int(x)))  # 정수로 변환
    y = max(0, min(volume_data.shape[1] - 1, int(y)))  # 정수로 변환
    z = max(0, min(volume_data.shape[2] - 1, int(z)))  # 정수로 변환

    result = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                coefficient = cubic_spline(x - i) * cubic_spline(y - j) * cubic_spline(z - k)
                result += coefficient * volume_data[x - i, y - j, z - k]

    return result

# 볼륨 데이터 (가정)
volume_data = np.random.rand(10, 10, 10)
print(volume_data)

# 보간 위치 (가정)
x, y, z = 5.3, 6.7, 7.1

# 트리큐빅 스플라인 보간
interpolated_value = tricubic_spline_interpolation(volume_data, x, y, z)
print("Interpolated Value:", interpolated_value)
