import numpy as np, cv2, pickle

def findCorners(image, bSize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, bSize) # 코너 검출

    if ret:        # 부화소(subpixel) 위치를 찾아서 코너 좌표 개선
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 30, 0.1)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, np.array(corners, np.float32), image

def show_image(file, bSize, result):
    cv2.drawChessboardCorners(result[2], bSize, result[1], result[0])  # 코너 표시
    cv2.imshow(file, result[2])

def calibrate_correct(objectPoints, imagePoints, image):
    size = image.shape[1::-1]
    ret = cv2.calibrateCamera(objectPoints, imagePoints, size, None, None)

    newSacle, roi = cv2.getOptimalNewCameraMatrix(ret[1], ret[2], size, 1)
    undistorted = cv2.undistort(image, ret[1], ret[2], None, newSacle)
    x, y, w, h = roi
    return ret, undistorted, undistorted[y:y + h, x:x + w]  # 왜곡 영역 제거

bSize = (8, 7)                                      # 체스판 코너 개수(가로, 세로)
points = [(x, y, 0) for y in range(bSize[1]) for x in range(bSize[0])]
points = np.array(points, np.float32)
print(points.shape)

# 코너 좌표 및 실세계 3차원 좌표
files = ["chessboard_01", "chessboard_02", "chessboard_02"]
images = [cv2.imread('images/%s.jpg' % file, 1) for file in files]
results = [findCorners(image, bSize) for image in images]
imagePoints = [result[1] for result in results if result[0]]
objectPoints = [points] * len(imagePoints)

[show_image(f, bSize, result) for f, result in zip(files, results) if result[0]]

image = cv2.imread("images/chessboard_05.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 에러")

ret, undistorted, correct_img = calibrate_correct(objectPoints, imagePoints, image)

print("RMS error reported by cv2.calibrateCamera:", ret[0])
print("cameraMatrix =\n%s" % ret[1])
print("distCoeffs =\n%s" % ret[2])
print("rvecs =\n%s" % np.reshape(ret[3], (3,-1)))
print("tvecs =\n%s" % np.reshape(ret[4], (3,-1)))

with open('camera_calibration.txt', 'wb') as f:
    pickle.dump(ret, f)

cv2.imshow("original", image)
cv2.imshow("undistorted", undistorted)
cv2.imshow("cropUndistorted", correct_img)
cv2.waitKey(0)