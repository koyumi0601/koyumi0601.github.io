import numpy as np
import cv2

image = np.zeros((300, 400), np.uint8)
image.fill(200)                     # 혹은 image[:] = 200

cv2.imshow("Window title", image)
cv2.waitKey(0)
cv2.destroyAllWindows()