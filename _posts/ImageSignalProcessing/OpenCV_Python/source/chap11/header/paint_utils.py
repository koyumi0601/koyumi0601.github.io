import numpy as np, cv2

def place_icons(image, size):
    icon_name = ["rect" , "circle", "eclipe", "line",   # 아이콘 파일 이름
                 "brush", "eraser", "open"  , "save",
                 "plus" , "minus" , "clear" , "color"]

    icons = [(i%2, i//2, 1, 1) for i in range(len(icon_name))]
    icons = np.multiply(icons, size*2)                  # icons 모든 원소에 size 곱합

    for roi, name in zip(icons, icon_name):
        icon = cv2.imread('images/icon/%s.jpg' %name , cv2.IMREAD_COLOR)
        if icon is None: continue
        x, y, w, h = roi
        image[y:y+h, x:x+w] = cv2.resize(icon, size)
    return list(icons)                   # 팔레트 생성

def create_hueIndex(image, roi):
    x, y, w, h =  roi                         # 관심영역 너비, 높이
    index = [[(j, 1, 1) for j in range(w)] for i in range(h)]      # 가로로 만들기
    ratios = (180 / w, 255, 255)
    hueIndex = np.multiply(index, ratios).astype('uint8')  # HSV 화소값 행렬

    image[y:y+h, x:x+w] = cv2.cvtColor(hueIndex, cv2.COLOR_HSV2BGR)

def create_colorPlatte(image, idx, roi):
    x, y, w, h = roi
    hue = idx-x
    palette = [[(hue, j, h-i-1) for j in range(w)] for i in range(h)]
    ratios = (180/w, 255/w, 255/h )
    palette = np.multiply(palette, ratios).astype('uint8')

    image[y:y+h, x:x+w] = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)

def create_colorPlatte1(image, hueidx, roi):
    x, y, w, h = roi
    ratio1 = 180 / h                     # 팔레트 높이에 따른 색상 비율
    ratio2 = 255 / w                      # 팔레트 너비에 따른 채도 비율
    ratio3 = 255 / h                     # 팔레트 높이에 따른 명도 비율
    hue = ((hueidx - x) * ratio1)           # 색상팔레트 기본 색상

    palette = [[(hue, j, (h-i-1)) for j in range(w)] for i in range(h)]
    palette = np.multiply(palette, (1, ratio2, ratio3)).astype('uint8')

    image[y:y+h, x:x+w] = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)

def create_colorPlatte2(image, hueidx, roi):
    x, y, w, h = roi
    ratio1 = 180 / w  # 팔레트 높이에 따른 색상 비율
    ratio2 = 256 / w  # 팔레트 너비에 따른 채도 비율
    ratio3 = 256 / h  # 팔레트 높이에 따른 명도 비율

    hue = round((hueidx - x) * ratio1)  # 색상 팔레트 기본 색상
    palette = [[(hue, j * ratio2, (h - i - 1) * ratio3)  # (색상, 채도, 명도) 화소 구성
                for j in range(w)] for i in range(h)]  # roi 크기 순회
    palette = np.array(palette, np.uint8)

    image[y:y + h, x:x + w] = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)

    return hue
