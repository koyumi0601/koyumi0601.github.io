from coin_preprocess import *

image, th_img = preprocessing(70)                           # 전처리 수행
if image is None: raise Exception("영상 파일 읽기 에러")

circles = find_coins(th_img)                            # 객체(회전사각형) 검출
for center, radius in circles:
    cv2.circle(image, center, radius, (0, 255, 0), 2)   # 동전 영상 원으로 표시

cv2.imshow("preprocessed image", th_img)
cv2.imshow("coin image", image)
cv2.waitKey(0)