import numpy as np, cv2, math

def calc_hsi(bgr):
    # B, G, R = bgr.astype(float)                           # float 형 변환
    B, G, R = float(bgr[0]), float(bgr[1]), float(bgr[2])       # 속도면에 유리
    bgr_sum = (R + G + B)
    # 색상 계산
    tmp1 = ((R - G) + (R - B)) * 0.5
    tmp2 = math.sqrt((R - G) * (R - G) + (R - B) * (G - B))
    angle = math.acos(tmp1 / tmp2) * (180 / np.pi) if tmp2 else 0

    H = angle if B <= G else 360 - angle
    S = 1.0 - 3 * min([R, G, B]) / bgr_sum if bgr_sum else 0
    I = bgr_sum / 3                                                # 명도 계산
    return (H/2, S*255, I)

# BGR 컬러 -> HSI 컬러
def bgr2hsi(image):
    hsv = [[calc_hsi(pixel) for pixel in row] for row in image ]   # 2차원 배열 순회
    return (np.array(hsv)).astype('uint8')

BGR_img = cv2.imread("images/color_space.jpg", cv2.IMREAD_COLOR) # 컬러 영상 읽기
if BGR_img is None: raise Exception("영상 파일 읽기 오류")

HSI_img = bgr2hsi(BGR_img)                  # BGR를 HSI로 변환
HSV_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2HSV) # OpenCV 함수
Hue, Saturation, Intensity = cv2.split(HSI_img)                    # 채널 분리
Hue2, Saturation2, Intensity2 = cv2.split(HSV_img) 					# 채널 분리

titles = ['BGR_img','Hue','Saturation','Intensity']
[cv2.imshow(t, eval(t)) for t in titles]
[cv2.imshow('OpenCV_'+t, eval(t+'2')) for t in titles[1:]]	# OpenCV 결과 영상 표시
cv2.waitKey(0)