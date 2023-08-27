import numpy as np, cv2

def make_palette(rows):
    # 리스트 생성 방식
    # hue = [round(i * 180 / rows) for i in range(rows)]  # hue 값 리스트 계산
    hue = np.arange(0, rows ) * 180 / rows
    hsv = [[(h, 255, 255)] for h in hue]                # (hue, 255,255) 화소값 계산
    hsv = np.array(hsv, np.uint8)                       # numpy 행렬의 uint8형 변환
    # # 반복문 방식
    # hsv = np.full((rows, 1, 3), (255,255,255), np.uint8)
    # for i in range(0, rows):                                # 행수만큼 반복
    #     hue = round(i / rows * 180 )                        # 색상 계산
    #     hsv[i] = (hue, 255, 255)                            # HSV 컬러 지정
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)         # HSV 컬러 -> BGR 컬러

def draw_histo_hue(hist, shape=(200, 256,3)):
    hsv_palate = make_palette(hist.shape[0])                      # 색상 팔레트 생성
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)    # 정규화

    gap = hist_img.shape[1] / hist.shape[0]  # 한 계급 크기
    for i, h in enumerate(hist):
        x, w = int(round(i * gap)), int(round(gap))
        color = tuple(map(int, hsv_palate[i][0]))                    # 정수형 튜플로 변환
        cv2.rectangle(hist_img, (x,0,w,int(h)), color , cv2.FILLED) # 팔레트 색으로 그리기

    return cv2.flip(hist_img, 0)

image = cv2.imread("images/hue_hist.jpg", cv2.IMREAD_COLOR)  # 영상 읽기
if image is None: raise Exception("영상 파일 읽기 오류")

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   # BGR 컬러 -> HSV 컬러     
hue_hist = cv2.calcHist( [hsv_img], [0], None, [18], [0,180])       # Hue 채널 히스토그램 계산
hue_hist_img = draw_histo_hue(hue_hist, (200, 360, 3)) # 히스토그램 그래프

cv2.imshow("image", image)
cv2.imshow("hue_hist_img", hue_hist_img)
cv2.waitKey(0)