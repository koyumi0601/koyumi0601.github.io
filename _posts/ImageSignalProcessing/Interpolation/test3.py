# uni-variant
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy
import natsort
import time

def cubic_spline_interpolation_1D(x, value, x_new):
    x = np.array(x).astype(float)
    value = np.array(value).astype(float)
    variableNum = 4 # because it is cubic
    diagonalMtrRowLength = int(variableNum * (len(x) - 1))
    solvor_input_value = []
    for idx in range(len(value)):
        if idx == 0 or idx == len(value)-1:
            solvor_input_value.append(value[idx]) # first boundary knot and last boundary knot
            solvor_input_value.append(0.0) # first and last curve as "Not-a-knot"
        else:
            solvor_input_value.append(value[idx]) # knot for lefthand
            solvor_input_value.append(value[idx]) # knot for righthand
            solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0
            solvor_input_value.append(0.0)    # Second derivative of lefthand and righthand, then its difference equal to 0
    solvor_input_value = np.array(solvor_input_value)
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
    w = np.linalg.solve(solvor_input_Mtx, solvor_input_value)
    for idx in range(int(len(x)-1)):
        if idx == 0:
            tmp_x_new = x_new[(x_new >= x[idx]) & (x_new <= x[idx+1])]
            value_new = np.polyval(w[int(idx)*variableNum:int(idx+1)*variableNum], tmp_x_new)
        else:
            tmp_x_new = x_new[(x_new > x[idx]) & (x_new <= x[idx+1])]
            value_new = np.concatenate((value_new, np.polyval(w[int(idx)*variableNum:int(idx+1)*variableNum], tmp_x_new)), 0)
    return value_new


def cubic_spline_interpolation_3D(x, y, z, value, x_new, y_new, z_new):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    z = np.array(z).astype(float)
    value = np.array(value).astype(float)
    variableNum = 20 # because it is tricubic
    diagonalMtrRowLength = int(variableNum * (len(x) - 1) * (len(y) - 1) * (len(z) - 1))
    print(f'diagonalMtrRowLength: {diagonalMtrRowLength}') 

    solvor_input_value = []
    for idx_x in range(len(x)):
        print(f'len(x): {len(x), len(y), len(z)}')
        for idx_y in range(len(y)):
            
            for idx_z in range(len(z)):
                # 7 6 5 4 3 2 1 0
                if (idx_x == 0 or idx_x == len(x)-1) and (idx_y == 0 or idx_y == len(y)-1) and (idx_z == 0 or idx_z == len(z)-1): # corner of x, y, z
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left or right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up or down
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front or rear    
                if (idx_x == 0 or idx_x == len(x)-1) and (idx_y == 0 or idx_y == len(y)-1) and not(idx_z == 0 or idx_z == len(z)-1): # corner of x, y
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock rear
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left or right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up or down
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, rear    
                if (idx_x == 0 or idx_x == len(x)-1) and not(idx_y == 0 or idx_y == len(y)-1) and (idx_z == 0 or idx_z == len(z)-1): # corner of x, z
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock up
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock down
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left or right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, down    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front or rear    
                if (idx_x == 0 or idx_x == len(x)-1) and not(idx_y == 0 or idx_y == len(y)-1) and not(idx_z == 0 or idx_z == len(z)-1): # corner of x
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock up, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock up, rear
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock down, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock down, rear
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left or right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, down    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, rear                    
                if not(idx_x == 0 or idx_x == len(x)-1) and (idx_y == 0 or idx_y == len(y)-1) and (idx_z == 0 or idx_z == len(z)-1): # corner of y, z
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock left
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up or down    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front or rear    
                if not(idx_x == 0 or idx_x == len(x)-1) and (idx_y == 0 or idx_y == len(y)-1) and not(idx_z == 0 or idx_z == len(z)-1): # corner of y
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock left, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock left, rear
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock right, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock right, rear
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up or down    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, rear                    
                if not(idx_x == 0 or idx_x == len(x)-1) and not(idx_y == 0 or idx_y == len(y)-1) and (idx_z == 0 or idx_z == len(z)-1): # corner of z
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock left, up
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock left, down
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock right, up
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # first boundary knot and last boundary knot, subblock right, down
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, left
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" x, right
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, up    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" y, down    
                    solvor_input_value.append(0.0) # first and last curve as "Not-a-knot" z, front or rear 
                if not(idx_x == 0 or idx_x == len(x)-1) and not(idx_y == 0 or idx_y == len(y)-1) and not(idx_z == 0 or idx_z == len(z)-1): # not corner
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for left, up, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for left, up, rear
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for left, down, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for left, down, rear
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for right, up, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for right, up, rear
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for right, down, front
                    solvor_input_value.append(value[idx_x, idx_y, idx_z]) # knot for right, down, rear        
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 x, left
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 x, right
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 y, up
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 y, down
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 z, front
                    solvor_input_value.append(0.0)    # first derivative of lefthand and righthand, then its difference equal to 0 z, rear
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 x, left
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 x, right
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 y, up
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 y, down
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 z, front
                    solvor_input_value.append(0.0)    # second derivative of lefthand and righthand, then its difference equal to 0 z, rear



    solvor_input_value = np.array(solvor_input_value)
    print(f'solvor_input_value: {solvor_input_value.shape}')

    solvor_input_value = []
    matrix = [[] ,[] , [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    print('hello')
    pass


if __name__ == "__main__":
    # data
    # x = np.array([0.05, 1, 3, 5, 6])
    # value = np.array([0, 1, 0, -1, 0])
    test_dim = 4
    x = np.random.rand(test_dim)
    y = np.random.rand(test_dim)
    z = np.random.rand(test_dim)
    value = np.random.rand(test_dim, test_dim, test_dim)
    sortedIndice_x = np.argsort(x)
    sortedIndice_y = np.argsort(y)
    sortedIndice_z = np.argsort(z)
    x = x[sortedIndice_x]
    y = y[sortedIndice_y]
    z = z[sortedIndice_z]
    # value = value[sortedIndice]

    x_new = np.linspace(np.min(x), np.max(x), 100)
    y_new = np.linspace(np.min(y), np.max(y), 100)
    z_new = np.linspace(np.min(z), np.max(z), 100)
    
    cubic_spline_interpolation_3D(x, y, z, value, x_new, y_new, z_new)
    abcd

    # 
    # library
    cs = interp1d(x, value, kind='cubic')
    st = time.time()
    y_new1 = cs(x_new)
    print(time.time()-st)

    # solver
    st = time.time()
    y_new2 = cubic_spline_interpolation_1D(x, value, x_new)
    print(time.time()-st)

    plt.plot(x, value, 'o', label='Data points')
    plt.plot(x_new, y_new1, "-b", linewidth=2, label='Cubic Spline interpolation by the proven library')
    plt.plot(x_new, y_new2, "--r", linewidth=1, label='Cubic Spline interpolation by myself-implementation')
    plt.legend()
    plt.title(f"Abs diff between library and implented is {np.sum(np.abs(y_new1 - y_new2))}({np.round(np.sum(np.abs(y_new1 - y_new2)) / np.sum(np.abs(y_new1)) * 100, 2)}%)")
    plt.grid()
    plt.show()
