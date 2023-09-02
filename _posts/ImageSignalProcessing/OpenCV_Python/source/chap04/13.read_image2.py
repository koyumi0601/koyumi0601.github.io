import cv2
from Common.utils import print_matInfo  # 행렬 정보 출력 함수 임포트

title1, title2 = "color2gray", "color2color"
color2gray = cv2.imread("images/read_color.jpg", cv2.IMREAD_GRAYSCALE)
color2color = cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
if color2gray is None or color2color is None:
    raise Exception("영상 파일 읽기 에러")

print("행렬 좌표 (100, 100) 화소값")
print("%s %s" % (title1, color2gray[100, 100]))     # 한 화소값 표시
print("%s %s\n" % (title2, color2color[100, 100]))

print_matInfo(title1, color2gray)                   # 행렬 정보 출력
print_matInfo(title2, color2color)
cv2.imshow(title1, color2gray)                      # 행렬 정보 영상으로 띄우기
cv2.imshow(title2, color2color)
cv2.waitKey(0)