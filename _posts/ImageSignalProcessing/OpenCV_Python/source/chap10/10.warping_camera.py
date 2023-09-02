import numpy as np, cv2
from Common.utils import put_string
from Common.calibration import *

bSize, mode = (8,7), 'detect mode'             # 체스판 코너수, 카메라 모드
imagePoints= []                                 # 영상 코너 좌표
points = [(x, y, 0) for y in range(bSize[1]) for x in range(bSize[0])]
points = np.array(points, np.float32)

capture = cv2.VideoCapture(0)
if capture.isOpened() is False: raise Exception('카메라 연결 안됨')
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)
print('카메라 연결 완료')

while(True):
    ret, frame = capture.read()
    key = cv2.waitKey(30)
    if key == 27 or ret is False : break     #  esc 키이면 종료
    if key == 13: mode = 'correct mode'      # 엔터키
    if key == 8:  mode = 'detect mode'       # 백페이스바키

    if mode =='detect mode':
        ret, corners, img = findCorners(frame, bSize)    # 코너 검출
        if ret: show_image('image', bSize, (ret, corners, img))

        if ret and key == 32:                      # 스페이스바이면 저장
            imagePoints.append(corners)             # 코너 좌표 저장
            put_string(frame, "save cord: ", (10, 40), len(imagePoints) )
            cv2.imshow("image", frame)
            cv2.waitKey(500)                        # 표시 내용 보이기 위해 잠깐 멈춤
        else:
            put_string(frame, mode, (10, 40), '')
            cv2.imshow("image", frame)

    elif mode == 'correct mode':
        if len(imagePoints) >= 3 :
            objectPoints = [points] * len(imagePoints)
            _, _, correct_img = calibrate_correct(objectPoints, imagePoints, frame)
            put_string(frame, mode, (400, 40), '')
            cv2.imshow('image', frame)
        else:
            put_string(frame, "Capture more than 3 corner coordinates", (70, 200))
            cv2.imshow("image", frame)
            cv2.waitKey(1000)
            mode = 'detect mode'

capture.release()