import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy
import natsort
import time

def cubic_spline_interpolation_1D(x, y, x_new):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    variableNum = 4 # because it is cubic
    diagonalMtrRowLength = int(variableNum * (len(x) - 1))
    solvor_input_y = []
    for idx in range(len(y)):
        if idx == 0 or idx == len(y)-1:
            solvor_input_y.append(y[idx]) # first boundary knot and last boundary knot
            solvor_input_y.append(0.0) # first and last curve as "Not-a-knot"
        else:
            solvor_input_y.append(y[idx]) # knot for lefthand
            solvor_input_y.append(y[idx]) # knot for righthand
            solvor_input_y.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0
            solvor_input_y.append(0.0)    # Second derivative of lefthand and righthand, then its difference equal to 0
    solvor_input_y = np.array(solvor_input_y)
    solvor_input_Mtx = []
    for idx in range(len(x)):
        if idx == 0:
            # first boundary knot
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([1.0, 1.0, 1.0, 1.0]) * np.array([x[idx]**3, x[idx]**2, x[idx]**1, x[idx]**0])
            row[int(idx)*variableNum:int(idx+1)*variableNum] = tmp
            solvor_input_Mtx.append(row)
            # first curve as "Not-a-knot" (third derivative of first curve and second curve, then its difference equal to 0)
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([6.0, 0.0, 0.0, 0.0]) * np.array([x[idx]**1, 0, 0, 0])
            row[int(idx)*variableNum:int(idx+1)*variableNum] = tmp
            row[int(idx+1)*variableNum:int(idx+2)*variableNum] = -tmp
            solvor_input_Mtx.append(row)
        elif idx == len(x)-1:
            # last boundary knot
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([1.0, 1.0, 1.0, 1.0]) * np.array([x[idx]**3, x[idx]**2, x[idx]**1, x[idx]**0])
            row[int(idx-1)*variableNum:int(idx)*variableNum] = tmp
            solvor_input_Mtx.append(row)
            # last curve as "Not-a-knot" (third derivative of last curve and second last curve, then its difference equal to 0)
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([6.0, 0.0, 0.0, 0.0]) * np.array([x[idx]**1, 0, 0, 0])
            row[int(idx-1)*variableNum:int(idx)*variableNum] = tmp
            row[int(idx-2)*variableNum:int(idx-1)*variableNum] = -tmp
            solvor_input_Mtx.append(row)
        else:
            # knot for lefthand
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([1.0, 1.0, 1.0, 1.0]) * np.array([x[idx]**3, x[idx]**2, x[idx]**1, x[idx]**0])
            row[int(idx-1)*variableNum:int(idx)*variableNum] = tmp
            solvor_input_Mtx.append(row)
            # knot for righthand
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([1.0, 1.0, 1.0, 1.0]) * np.array([x[idx]**3, x[idx]**2, x[idx]**1, x[idx]**0])
            row[int(idx)*variableNum:int(idx+1)*variableNum] = tmp
            solvor_input_Mtx.append(row)
            # first derivative of lefthand and righthand, then its difference equal to 0
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([3.0, 2.0, 1.0, 0.0]) * np.array([x[idx]**2, x[idx]**1, x[idx]**0, 0])
            row[int(idx-1)*variableNum:int(idx)*variableNum] = tmp
            row[int(idx)*variableNum:int(idx+1)*variableNum] = -tmp
            solvor_input_Mtx.append(row)
            # Second derivative of lefthand and righthand, then its difference equal to 0
            row = np.zeros((diagonalMtrRowLength))
            tmp = np.array([6.0, 2.0, 0.0, 0.0]) * np.array([x[idx]**1, x[idx]**0, 0, 0])
            row[int(idx-1)*variableNum:int(idx)*variableNum] = tmp
            row[int(idx)*variableNum:int(idx+1)*variableNum] = -tmp
            solvor_input_Mtx.append(row)
    solvor_input_Mtx = np.array(solvor_input_Mtx)
    w = np.linalg.solve(solvor_input_Mtx, solvor_input_y)
    for idx in range(int(len(x)-1)):
        if idx == 0:
            tmp_x_new = x_new[(x_new >= x[idx]) & (x_new <= x[idx+1])]
            y_new = np.polyval(w[int(idx)*variableNum:int(idx+1)*variableNum], tmp_x_new)
        else:
            tmp_x_new = x_new[(x_new > x[idx]) & (x_new <= x[idx+1])]
            y_new = np.concatenate((y_new, np.polyval(w[int(idx)*variableNum:int(idx+1)*variableNum], tmp_x_new)), 0)
    return y_new


if __name__ == "__main__":
    # x = np.array([0.05, 1, 3, 5, 6])
    # y = np.array([0, 1, 0, -1, 0])
    x = np.random.rand(30)
    y = np.random.rand(30)
    sortedIndice = np.argsort(x)
    x = x[sortedIndice]
    y = y[sortedIndice]

    x_new = np.linspace(np.min(x), np.max(x), 1000)

    # library
    cs = interp1d(x, y, kind='cubic')
    st = time.time()
    y_new1 = cs(x_new)
    print(time.time()-st)

    # solver
    st = time.time()
    y_new2 = cubic_spline_interpolation_1D(x, y, x_new)
    print(time.time()-st)

    plt.plot(x, y, 'o', label='Data points')
    plt.plot(x_new, y_new1, "-b", linewidth=2, label='Cubic Spline interpolation by the proven library')
    plt.plot(x_new, y_new2, "--r", linewidth=1, label='Cubic Spline interpolation by myself-implementation')
    plt.legend()
    plt.title(f"Abs diff between library and implented is {np.sum(np.abs(y_new1 - y_new2))}({np.round(np.sum(np.abs(y_new1 - y_new2)) / np.sum(np.abs(y_new1)) * 100, 2)}%)")
    plt.grid()
    plt.show()
