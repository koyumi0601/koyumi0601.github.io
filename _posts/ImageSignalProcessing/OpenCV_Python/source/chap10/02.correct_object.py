import numpy as np, cv2
from Common.hough import *                    # 허프 변환 관련 사용자 정의 함수 포함

def detect_maxObject(img):
    # 외곽선 검출 - Opnecv 4.0부터 반환값은 2개 원소 갖는 튜플
    results = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if int(cv2.__version__[0]) >= 4:                # Opnecv 4.0은 2원소 튜플 반환
        contours = results[0]
    else:
        contours = results[1]				# OpenCV 3.x은 3원소 튜플 반환

    areas = [cv2.contourArea(c) for c in contours]
    idx = np.argsort(areas)
    max_rect = contours[idx[-1]]

    rect = cv2.boundingRect(max_rect)        # 외곽선을 모두 포함하는 사각형 반환
    rect = np.add(rect, (-10, -10, 20, 20))   # 검출 객체 사각형 크기 확대
    return rect

image = cv2.imread('images/harness.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 에러")
rho, theta = 1, np.pi / 180                             # 허프변환 거리간격, 각도간격
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 명암도 영상 변환
_, th_gray = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) # 이진 영상 변환
kernel = np.ones((3, 3), np.uint8)
morph = cv2.erode(th_gray, kernel, iterations=2)        # 침식 연산 - 2번 반복

x, y, w, h = detect_maxObject(np.copy(morph))               # 가장 큰 객체 검출
roi = th_gray[y:y+h, x:x+w]

canny = cv2.Canny(roi, 40, 100)                         # 캐니 에지 검출
lines = houghLines(canny, rho, theta, 50)               # 허프 직선 검출
# lines = cv2.HoughLines(canny, rho, theta, 50)         # OpenCV 함수

cv2.rectangle(morph, (x, y, w, h), 100, 2)                   # 큰 객체 사각형 표시
canny_line = draw_houghLines(canny, lines, 1)           # 직선 표시

angle = (np.pi - lines[0, 0, 1]) * 180 / np.pi           # 회전 각도 계산
h, w = image.shape[:2]
center = (w//2, h//2)                           # 입력 영상의 중심점
rot_map = cv2.getRotationMatrix2D(center, -angle, 1)    # 반대방향 회전 행렬 계산
dst = cv2.warpAffine(image, rot_map, (w, h), cv2.INTER_LINEAR)  # 역회전 수행

cv2.imshow("image", image)
cv2.imshow("morph", morph)
cv2.imshow("line_detect", canny_line)
cv2.resizeWindow("line_detect", 250, canny_line.shape[0])
cv2.imshow("dst", dst)
cv2.waitKey(0)