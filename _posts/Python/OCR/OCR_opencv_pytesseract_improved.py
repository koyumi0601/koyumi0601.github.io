import cv2
import numpy as np
import pytesseract

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지 파일 로드
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\image_00249.png'
# image_00028.png
# image_00039.png
# image_00172.png
# image_00249.png
image = cv2.imread(image_path)

# 읽고자 하는 이미지의 특정 영역을 정의
x1 = 1700
y1 = 268
x2 = 1788
y2 = 290
roi_cropped = image[y1:y2, x1:x2]

# 추출한 영역을 Grayscale로 변환
roi_gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)

# 이미지의 밝기가 일정 임계값 이상이면 반전
if np.mean(roi_gray) > 127:
    max_val = 191
    min_val = 16
    print('revert')
    roi_gray = cv2.bitwise_not(roi_gray)
    roi_gray = max_val + min_val - roi_gray


# Otsu의 이진화
ret, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 모폴로지 연산으로 노이즈 제거
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
processed_roi = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Tesseract로 전처리된 이미지 영역에서 텍스트를 인식
extracted_text = pytesseract.image_to_string(processed_roi, lang='eng')

# 추출한 텍스트 출력
print("인식된 텍스트:", extracted_text)

# 결과 이미지 표시 (선택적, 개발 환경에 따라 주석 처리할 것)
# cv2.imshow('Processed ROI', processed_roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()