# 필요한 라이브러리를 설치합니다. 이미 설치되어 있다면 이 부분은 건너뛰세요.
# !pip install easyocr
from PIL import Image
import easyocr
import numpy as np

# OCR Reader를 초기화합니다. 여기서 'en'은 영어를 의미합니다.
# 숫자 인식만 필요한 경우 'en'을 사용해도 충분합니다.
reader = easyocr.Reader(['en'])

# 추출하고자 하는 이미지의 경로를 지정합니다.

image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.png'
image = Image.open(image_path)

# 관심 영역(coordinates) 설정: (x1, y1, x2, y2)
# 여기서 (x1, y1)은 영역의 왼쪽 상단 모서리이고, (x2, y2)는 오른쪽 하단 모서리입니다.
# (1649, 268, 1788, 290)
x1 = 1649
x1 = 1700
y1 = 268
x2 = 1788
y2 = 290
coordinates = (x1, y1, x2, y2)

# 이미지에서 관심 영역만 잘라내기
cropped_image = image.crop(coordinates)

# 잘라낸 이미지를 OCR로 텍스트 추출하기
reader = easyocr.Reader(['en'], gpu=True)  # 여기서 사용할 언어와 GPU 사용 여부를 설정합니다.
result = reader.readtext(np.array(cropped_image))



# 결과 출력
for detection in result:
    text = detection[1]
    print(text)





# # 결과를 출력합니다.
# for (bbox, text, prob) in results:
#     # bbox는 텍스트의 경계 상자(bounding box), text는 인식된 텍스트, prob는 확률을 의미합니다.
#     print(f'Detected text: {text} (Confidence: {prob:.2f})')

# # 필요에 따라 특정 형식의 데이터만 필터링하여 추출할 수 있습니다.
# # 예를 들어, 숫자와 점(.)만 추출하려면 아래와 같이 할 수 있습니다.
# import re

# for (bbox, text, prob) in results:
#     if re.fullmatch(r'[0-9.]+', text):  # 정규 표현식을 사용하여 숫자와 점(.)만 매치합니다.
#         print(f'Number detected: {text} (Confidence: {prob:.2f})')