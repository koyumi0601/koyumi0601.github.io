# 필요한 라이브러리 불러오기
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import time
import numpy as np

# 태서랙트 실행 파일 경로 설정 (Windows 환경에서 필요함)
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_from_cropped_area(image_path):
    """
    이미지에서 주어진 좌표 영역의 텍스트를 OCR로 추출하는 함수
    
    Parameters:
    - image_path (str): 이미지 파일 경로
    - coordinates (tuple): 잘라낼 영역의 좌표 (x1, y1, x2, y2)
    
    Returns:
    - text (str): 추출된 텍스트
    """

    offset_x = 100
    offset_y = 100
    # coordinates = (92, 683, 160, 718)
    coordinates_full = (0, 0, 1920, 1080)
    coordinates_IPDA_p1 = (1648, 85, 1789, 340)
    coordinates = (1648, 85, 1789, 340)

    coordinates_IPDA = (1649, 86, 1788, 339)
    coordinates_depth = (1649, 268, 1788, 290)
    # coordinates = (1649, 268, 1788, 290)

    # coordinates = (1630, 268, 1879, 290)
    # coordinates = (1720-offset_x, 268-offset_y, 1785+offset_x, 290+offset_y)

    image = Image.open(image_path)  # 이미지 불러오기
    image = image.convert('L') # grayscale
    # text = pytesseract.image_to_string(image, lang='eng')  # 잘라낸 이미지 영역에서 텍스트 추출하기
    # print(text)    
    image_array = np.array(image) # array. switch x, y
    print((image_array.shape))
    image_depth_array = image_array[268:290, 1649:1788]

    
    # image_array[268:290, 1649:1788] = 191 + 16 - image_array[268:290, 1649:1788] # 색상반전
    image_IPDA_array = image_array[86:339, 1649:1788] # ipda
    image_IPDA_array = image_array[85:340, 1648:1789] # ipda + 1
    image_IPDA_again = Image.fromarray(np.uint8(image_IPDA_array))
    plt.imshow(image_IPDA_again)
    plt.axis('off')
    plt.show()

    # text = pytesseract.image_to_string(cropped_image, lang='eng')  # 잘라낸 이미지 영역에서 텍스트 추출하기
    text = pytesseract.image_to_string(image_IPDA_again, lang='eng')  # 잘라낸 이미지 영역에서 텍스트 추출하기
    print(text)


    # plt.imshow(pixels)  # 잘라낸 이미지 시각화 (필요한 경우 주석 해제)
    # plt.axis('off')  # 그래프의 좌표축 제거
    # plt.show()

    # cropped_image_IPDA = image.crop(coordinates_IPDA)  # 주어진 좌표로 이미지 잘라내기
    # cropped_image_IPDA = cropped_image_IPDA.convert('L') # gray scale
    # cropped_image_IPDA_array = np.array(cropped_image_IPDA)

    # pixels = image.load()
    # print(pixels[1710, 275])
    # pixels[1649:1788, 268:290] = 40
    # plt.imshow(pixels)  # 잘라낸 이미지 시각화 (필요한 경우 주석 해제)
    # plt.axis('off')  # 그래프의 좌표축 제거
    # plt.show()

    # pixels_IPDA = pixels(coordinates_IPDA)
    # pixels_depth = pixels[coordinates_depth]
    # pixels_depth_revert = 


    white = 191
    gray = 41
    black = 16

    # text = pytesseract.image_to_string(cropped_image, lang='eng')  # 잘라낸 이미지 영역에서 텍스트 추출하기
    # plt.imshow(cropped_image)  # 잘라낸 이미지 시각화 (필요한 경우 주석 해제)
    # plt.axis('off')  # 그래프의 좌표축 제거
    # plt.show()
    # return text

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
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.png'




start_time = time.time()  # 실행 시작 시간 기록

# 이미지에서 텍스트 추출 및 숫자만 추출
extracted_text = ocr_from_cropped_area(image_path)
# depthCm = float(extract_numbers_from_string(extracted_text))
# print(extracted_text)
# print(depthCm)

end_time = time.time()  # 실행 종료 시간 기록
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} sec")