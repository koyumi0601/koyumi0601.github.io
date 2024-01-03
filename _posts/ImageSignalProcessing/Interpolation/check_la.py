import numpy as np
A = [[3, 4], [6, 8]]
y = [0, 0]
w = np.linalg.solve(A, y)
print(w)