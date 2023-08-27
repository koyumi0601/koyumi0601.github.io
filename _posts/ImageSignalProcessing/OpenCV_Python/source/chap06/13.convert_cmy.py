import numpy as np, cv2

BGR_img = cv2.imread("images/color_model.jpg", cv2.IMREAD_COLOR) # 컬러 영상 읽기
if BGR_img is None: raise Exception("영상 파일 읽기 오류")
    
white = np.array([255, 255, 255], np.uint8)
CMY_img = white - BGR_img
Cyan, Magenta, Yellow = cv2.split(CMY_img) # 채널 분리

titles = ['BGR_img','CMY_img','Cyan','Magenta','Yellow']
[cv2.imshow(t, eval(t)) for t in titles]
cv2.waitKey(0)