import cv2
from haar_histogram import draw_ellipse
from Common.utils import put_string

# 성별 분류
def classify(image, hists, no):

    global cnt1, cnt2, avg1, avg2
    criteria1 = cv2.compareHist(hists[2], hists[3], cv2.HISTCMP_CORREL)     # 유사도 비교
    criteria2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)

    tmp = 0.12 if criteria1 > 0 else 0.2           # 1차 비교
    value = 0 if criteria2 > tmp else 1            # 2차 비교
    text = "Man" if value else "Woman"

    text = "{0:02d}.jpg: ".format(no) + text                        # 분류 결과 출력
    put_string(image, text, (10, 30), "")           # 영상 출력
    result = "유사도 [입술: %4.2f 머리: %4.2f]" %(criteria1, criteria2)
    print(text + " - " + result)                    # 콘솔창 출력

    return image

# 눈, 얼굴 중심점, 얼굴, 입술 타원 표시
def display(image, face_center, eyes_center, sub):

    cv2.circle(image, eyes_center[0], 10, (0, 255, 0), 2)	# 눈 표시
    cv2.circle(image, eyes_center[1], 10, (0, 255, 0), 2)
    cv2.circle(image, face_center, 3, (0, 0, 255), 2)	    # 얼굴 중심점 표시

    draw_ellipse(image, sub[2], (255, 100, 0), 2, 0.45)     # 입술 타원
    draw_ellipse(image, sub[3], (0, 0, 255), 2, 0.45)	    # 얼굴 타원

    return image
 