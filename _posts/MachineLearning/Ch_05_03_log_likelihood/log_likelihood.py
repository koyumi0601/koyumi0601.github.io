import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
q = np.linspace(0.01, 0.99, 100)  # 0과 1을 제외한 범위에서 값을 생성 (log 함수 때문에)
y1 = (q - 1)**2
y2 = -np.log(q)

# 그래프 플롯
plt.figure(figsize=(10, 6))
plt.plot(q, y1, label='(q-1)^2', color='blue')
plt.plot(q, y2, label='-log(q)', color='red')

# 레이블 및 타이틀 설정
plt.xlabel('q')
plt.ylabel('Value')
plt.title('Plot of (q-1)^2 and -log(q)')
plt.legend()
plt.grid(True)
plt.show()