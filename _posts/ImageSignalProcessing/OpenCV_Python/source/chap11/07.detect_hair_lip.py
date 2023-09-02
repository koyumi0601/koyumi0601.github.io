from haar_utils import *                            # 전처리 및 영역 검출 함수 임포트

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # 정면 검출기
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  # 눈 검출기
image, gray = preprocessing(34)  # 전처리
if image is None: raise Exception("영상 파일을 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100));  # 얼굴 검출
if faces.any() :
    x, y, w, h = faces[0].tolist()
    face_image = image[y:y+h, x:x+w]  # 얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))  # 눈 검출

    if len(eyes) == 2:
        face_center = (x + w//2, y + h//2)
        eye_centers  = [[x+ex+ew//2, y+ey+eh//2] for ex,ey,ew,eh in eyes]
        corr_image, corr_center = correct_image(image, face_center, eye_centers )  # 기울기 보정

        rois = detect_object(face_center, faces[0])  # 머리 및 입술영역 검출

        cv2.rectangle(corr_image, rois[0], (255, 0, 255), 2)
        cv2.rectangle(corr_image, rois[1], (255, 0, 255), 2)
        cv2.rectangle(corr_image, rois[2], (255, 0, 0), 2)
        cv2.circle(corr_image, tuple(corr_center[0]), 5, (0, 255, 0), 2)
        cv2.circle(corr_image, tuple(corr_center[1]), 5, (0, 255, 0), 2)
        cv2.circle(corr_image, face_center, 3, (0, 0, 255), 2)
        cv2.imshow("correct_image", corr_image)
    else:
        print("눈 미검출")
else:
    cv2.imshow("image", image)
cv2.waitKey(0)