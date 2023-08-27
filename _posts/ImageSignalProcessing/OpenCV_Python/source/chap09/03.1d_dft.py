import numpy as np, math
import matplotlib.pyplot as plt

def exp(knN):
    th = -2 * math.pi * knN
    return complex(math.cos(th), math.sin(th))              # 복소수 클래스

def dft(g):
    N = len(g)
    dst = [sum(g[n] * exp(k*n/N ) for n in range(N)) for k in range(N) ]
    return np.array(dst)

def idft(G):
    N = len(G)
    dst = [sum(g[n] * exp(-k*n/N) for n in range(N)) for k in range(N) ]
    return np.array(dst) / N

fmax = 1000      # 샘플링 주파수 1000 Hz: 최대주파수의 2배
dt = 1/fmax      # 샘플링 간격
t = np.arange(0, 1, dt)         # 시간 리스트

g1 = np.sin(2*np.pi*50*t)
g2 = np.sin(2*np.pi*120*t)
g3 = np.sin(2*np.pi*260*t)
g = g1 * 0.6 + g2 * 0.9 + g3 * 0.2          # 신호 합성

N = len(g)                          # 신호 길이
df = fmax/N                         #
f = np.arange(0, N, df)
G = dft(g) * dt                    # 사용자 정의 DFT
g2 = idft(G) / dt

plt.figure(figsize=(8,8))
plt.subplot(3,1,1), plt.plot(t[0:200], g[0:200]), plt.title('org signal')
plt.subplot(3,1,2), plt.plot(f, np.abs(G) ), plt.title('dft amplitude')
plt.subplot(3,1,3), plt.plot(t[0:200], g2[0:200]), plt.title('idft signal')
plt.tight_layout()
plt.show()