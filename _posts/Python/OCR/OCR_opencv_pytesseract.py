import cv2
import pytesseract

# Tesseract 실행 파일 경로 설정
# 예: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지 파일 로드
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.png'
image = cv2.imread(image_path)

# 읽고자 하는 이미지의 특정 영역을 정의 (x, y, width, height)
x1 = 1649
x1 = 1700
y1 = 268
x2 = 1788
y2 = 290
roi_cropped = image[y1:y2, x1:x2]

# 추출한 영역을 Grayscale로 변환
roi_gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)

# Tesseract로 추출한 영역에서 텍스트를 인식
extracted_text = pytesseract.image_to_string(image, lang='eng')

# 추출한 텍스트 출력
print("인식된 텍스트:", extracted_text)

# # 결과 이미지 표시 (선택적)
# cv2.imshow('ROI', roi_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()