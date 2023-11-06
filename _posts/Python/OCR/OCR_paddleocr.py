from paddleocr import PaddleOCR, draw_ocr

# PaddleOCR 객체 생성
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use_angle_cls=True로 설정하여 회전된 텍스트 감지

# 이미지에서 텍스트 인식
result = ocr.ocr('path_to_image.jpg', cls=True)

# 인식 결과 출력
for line in result:
    print(line)

# 결과를 이미지로 그리기 (시각화)
image_path = 'path_to_image.jpg'
image = cv2.imread(image_path)
boxes = [line[0] for line in result]  # 좌표 추출
txts = [line[1][0] for line in result]  # 텍스트 추출
scores = [line[1][1] for line in result]  # 신뢰도 추출

# 시각화
im_show = draw_ocr(image, boxes, txts, scores, font_path='path_to_font.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')