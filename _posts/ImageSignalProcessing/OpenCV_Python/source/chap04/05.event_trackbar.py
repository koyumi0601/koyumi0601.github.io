import numpy as np
import cv2

def onChange(value):												# 트랙바 콜백 함수
    global image                        	# 전역 변수 참조

    add_value = value - int(image[0][0])        	# 트랙바 값과 영상 화소값 차분
    print("추가 화소값:", add_value)

    image[:] = image + add_value            		# 행렬과 스칼라 덧셈 수행
    cv2.imshow(title, image)

image = np.zeros((300, 500), np.uint8)           	# 영상 생성

title = 'Trackbar Event'
cv2.imshow(title, image)

cv2.createTrackbar("Brightness", title, image[0][0], 255, onChange)	# 트랙바 콜백 함수 등록
cv2.waitKey(0)
cv2.destroyWindow(title)