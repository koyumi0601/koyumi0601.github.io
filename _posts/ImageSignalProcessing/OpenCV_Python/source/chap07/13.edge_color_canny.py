import cv2

def onTrackbar(th):																	# 트랙바 콜백 함수
	edge = cv2.GaussianBlur(gray, (5, 5), 0)         	# 가우시안 블러링
	edge = cv2.Canny(edge, th, th*2, 5)						# 캐니 에지 검출

	color_edge = cv2.copyTo(image, mask=edge)
	dst = cv2.hconcat([image, color_edge])
	cv2.imshow("color edge", dst)

image = cv2.imread("images/color_edge.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상파일 읽기 오류")

th = 50
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    	# 명암도 영상 변환
dst = cv2.hconcat([image, image])
cv2.imshow("color edge", dst)
cv2.createTrackbar("Canny th", "color edge", th, 150, onTrackbar)	# 콜백 함수 등록
onTrackbar(th)																					# 콜백 함수 첫 실행
cv2.waitKey(0)