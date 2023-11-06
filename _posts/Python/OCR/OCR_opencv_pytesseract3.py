import cv2
import numpy as np
import pytesseract

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지 파일 로드
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\image_00001.png'
image = cv2.imread(image_path)

# 이미지의 명암 대비를 개선
alpha = 1.5 # 대비 조정 계수
beta = -30  # 밝기 조정
contrasted = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)

# ROI 추출
x1, y1, x2, y2 = 1700, 268, 1788, 290
roi_cropped = contrasted[y1:y2, x1:x2]

# Grayscale로 변환
roi_gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)

# 배경이 밝으면 반전
# ROI 영역의 평균 밝기를 계산하여 이를 기준으로 반전할지 결정합니다.
if np.mean(roi_gray) > 127:  # 배경이 밝으면 (평균 밝기가 중간값 이상)
    roi_gray = cv2.bitwise_not(roi_gray)  # 반전

# 적응형 이진화 적용
adaptive_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Tesseract로 텍스트 인식
# PSM 모드 6은 이미지가 단일 균일한 블록의 텍스트를 포함한다고 가정합니다.
extracted_text = pytesseract.image_to_string(adaptive_thresh, lang='eng', config='--psm 6')

# 추출한 텍스트 출력
print("인식된 텍스트:", extracted_text)