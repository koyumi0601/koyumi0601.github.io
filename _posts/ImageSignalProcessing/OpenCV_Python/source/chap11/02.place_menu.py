from paint_init import *
from paint_utils import *

image = np.full((500, 800, 3), 255, np.uint8)
icons = place_icons(image, (60, 60))                # 아이콘 배치, 아이콘 크기
x, y, w, h = icons[-1]                               # 아이콘 사각형 마지막 원소
palatte_roi  = (0, y + h + 2  , 120, 120)           # 팔레트 ROI
hueIndex_roi = (0, y + h + 124, 120, 15)            # 색상인덱스 ROI

icons.append(palatte_roi)      # 팔레트 사각형 추가
icons.append(hueIndex_roi)  # 색상인덱스 사각형 추가

create_hueIndex(image, icons[HUE_IDX])               # 색상인텍스 생성
create_colorPlatte(image, 0, icons[PALETTE])        # 팔레트 생성

cv2.imshow("PaintCV", image)
cv2.waitKey(0)


