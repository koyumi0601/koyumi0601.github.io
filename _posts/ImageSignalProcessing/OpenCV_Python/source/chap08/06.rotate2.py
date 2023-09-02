import numpy as np, cv2
from Common.interpolation import rotate_pt

def calc_angle(pts):
    d1 = np.subtract(pts[1], pts[0]).astype(float)        # 두 좌표간 차분 계산
    d2 = np.subtract(pts[2], pts[0]).astype(float)
    angle1 = cv2.fastAtan2(d1[1], d1[0])  # 차분으로 각도 계산
    angle2 = cv2.fastAtan2(d2[1], d2[0])
    return (angle2 - angle1)  # 두 각도 간의 차분

def draw_point(x, y, color):
    pts.append([x,y])
    print("좌표:", len(pts), [x,y])
    cv2.circle(tmp, (x, y), 2, color, 2)  # 중심 좌표 표시
    cv2.imshow("image", tmp)

def onMouse(event, x, y, flags, param):
    global tmp, pts
    if (event == cv2.EVENT_LBUTTONUP and len(pts) == 0):   draw_point(x, y, (0,0,255))
    if (event == cv2.EVENT_LBUTTONDOWN and len(pts) == 1): draw_point(x, y, (0,255, 0))
    if (event == cv2.EVENT_LBUTTONUP and len(pts) == 2):   draw_point(x, y, (255,0, 0))

    if len(pts) == 3:
        angle = calc_angle(pts)  # 회전각 계산
        print("회전각 : %3.2f" % angle)
        dst = rotate_pt(image, angle, pts[0])  # 사용자 정의 함수 회전 수행
        cv2.imshow("image", dst)
        tmp = np.copy(image)                    # 임시 행렬 초기화
        pts = []

image = cv2.imread('images/rotate.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 파일을 읽기 에러")
tmp = np.copy(image)
pts = []

cv2.imshow("image", image)
cv2.setMouseCallback("image", onMouse, 0)
cv2.waitKey(0)