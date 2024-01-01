import numpy as np
A = [[0, 1], [3, 4]]
y = [4, 0]
w = np.linalg.solve(A, y)
print(w)