# pip install tensorflow
# pip install keras-ocr
import keras_ocr

# keras-ocr 파이프라인 구축, 이는 모델을 로드하고 이미지에 대해 실행
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지에서 텍스트 인식
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\Python\OCR\img.png'
images = [
    keras_ocr.tools.read(image_path)  # 이미지 파일 경로
]

# 이미지 리스트에 대해 예측 실행
prediction_groups = pipeline.recognize(images)

# 예측 결과 출력
for image, predictions in zip(images, prediction_groups):
    print('Predictions for image:', predictions)

    # 시각화
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions)