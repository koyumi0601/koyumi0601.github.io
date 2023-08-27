import numpy as np, cv2

def place_icons(image, size):
    icon_name = ["rect", "circle", "eclipe", "line", "brush", "eraser",  # 아이콘 파일 이름
                 "open", "save", "plus", "minus", "clear", "color"]
    icons = [(i%2, i//2, 1, 1) for i in range(len(icon_name))]
    icons = np.multiply(icons, size*2)                  # icons 모든 원소에 size 곱합

    for roi, name in zip(icons, icon_name):
        icon = cv2.imread('images/icon/%s.jpg' % name, cv2.IMREAD_COLOR)
        if icon is None: continue
        x, y, w, h = roi
        image[y:y+h, x:x+w] = cv2.resize(icon, size)
    return list(icons)

image = np.full((500, 800, 3), 255, np.uint8)
icons = place_icons(image, (60, 60))                # 아이콘 배치, 아이콘 크기
cv2.imshow("PaintCV", image)
cv2.waitKey(0)


