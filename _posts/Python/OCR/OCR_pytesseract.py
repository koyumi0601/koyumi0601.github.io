# 필요한 라이브러리 불러오기
from PIL import Image
import pytesseract
# import matplotlib.pyplot as plt
import time

# 태서랙트 실행 파일 경로 설정 (Windows 환경에서 필요함)
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_from_cropped_area(image_path, coordinates):
    """
    이미지에서 주어진 좌표 영역의 텍스트를 OCR로 추출하는 함수
    
    Parameters:
    - image_path (str): 이미지 파일 경로
    - coordinates (tuple): 잘라낼 영역의 좌표 (x1, y1, x2, y2)
    
    Returns:
    - text (str): 추출된 텍스트
    """
    image = Image.open(image_path)  # 이미지 불러오기
    cropped_image = image.crop(coordinates)  # 주어진 좌표로 이미지 잘라내기
    text = pytesseract.image_to_string(cropped_image, lang='eng')  # 잘라낸 이미지 영역에서 텍스트 추출하기
    # plt.imshow(cropped_image)  # 잘라낸 이미지 시각화 (필요한 경우 주석 해제)
    # plt.axis('off')  # 그래프의 좌표축 제거
    # plt.show()
    return text

def extract_numbers_from_string(s):
    """
    문자열에서 숫자만 추출하는 함수
    
    Parameters:
    - s (str): 입력 문자열
    
    Returns:
    - str: 숫자만 포함된 문자열
    """
    return ''.join([char for char in s if char.isdigit()])

# 이미지 파일 경로와 OCR로 텍스트를 추출할 좌표 설정
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.tif'
coordinates = (92, 683, 160, 718)

start_time = time.time()  # 실행 시작 시간 기록

# 이미지에서 텍스트 추출 및 숫자만 추출
extracted_text = ocr_from_cropped_area(image_path, coordinates)
depthCm = float(extract_numbers_from_string(extracted_text))
print(extracted_text)
print(depthCm)

end_time = time.time()  # 실행 종료 시간 기록
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} sec")