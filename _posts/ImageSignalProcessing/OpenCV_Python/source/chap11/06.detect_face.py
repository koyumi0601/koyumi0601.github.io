import cv2, numpy as np

def preprocessing(no):  # 검출 전처리
    image = cv2.imread('images/face/%2d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    return image, gray

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # 정면 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # 눈 검출기
image, gray = preprocessing(34)  # 전처리
if image is None: raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100));  # 얼굴 검출
if faces.any():
    x, y, w, h = faces[0]
    face_image = image[y:y + h, x:x + w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))  # 눈 검출 수행
    if len(eyes) == 2:  # 눈 사각형이 검출되면
        for ex, ey, ew, eh in eyes:
            center = (x + ex + ew // 2, y + ey + eh // 2)
            cv2.circle(image, center, 10, (0, 255, 0), 2)  # 눈 중심에 원 그리기
    else:
        print("눈 미검출")

    cv2.rectangle(image, faces[0], (255, 0, 0), 2)  # 얼굴 검출 사각형 그리기
    cv2.imshow("image", image)

else: print("얼굴 미검출")
cv2.waitKey(0)