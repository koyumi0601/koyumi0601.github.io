from haar_utils import *                   # 검출기 적재 및 전처리 함수
from haar_histogram import *                  # 히스토그램 비교 관련 함수

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # 정면 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # 눈 검출기
image, gray = preprocessing(34)  # 전처리
if image is None: raise Exception("영상 파일을 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100));  # 얼굴 검출
if faces.any():
    x, y, w, h = faces[0].tolist()
    face_image = image[y:y + h, x:x + w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))  # 눈 검출

    if len(eyes) == 2:
        face_center = (x + w // 2, y + h // 2)
        eye_centers = [(x + ex + ew // 2, y + ey + eh // 2) for ex, ey, ew, eh in eyes]
        corr_image, corr_center = correct_image(image, face_center, eye_centers)  # 기울기 보정

        rois = detect_object(face_center, faces[0])  # 머리 및 입술영역 검출
        masks = make_masks(rois, corr_image.shape[:2])  # 4개 마스크 생성
        sim = calc_histo(corr_image, rois, masks)  # 4개 히스토그램 생성

        print("입술-얼굴 유사도: %4.2f" % sim[0])
        print("윗-귀밑머리 유사도: %4.2f" % sim[1])
    else:
        print("눈 미검출")
else:
    print("얼굴 미검출")