from paint_init import *
from paint_utils import *

def command(mode):
    global icons, image, Color, hue

    if mode == PALETTE:  # 색상팔레트 영역 클릭 시
        Color = tuple(map(int, image[pt2[::-1]]))
        x, y, w, h = icons[COLOR]
        image[y:y + h - 1, x:x + w - 1] = Color

    elif mode == HUE_IDX:                               # 색상인텍스 클릭 시
        create_colorPlatte(image, pt2[0], icons[PALETTE])  # 팔레트 새로 그리기

    cv2.imshow("paintCV", image)

def onMouse(event, x, y, flags, param):
    global pt1, pt2, mouse_mode, draw_mode

    if event == cv2.EVENT_LBUTTONUP:                # 왼쪽 버튼 떼기
        for i, (x0, y0, w, h) in enumerate(icons):  # 메뉴아이콘 사각형 조회
            if x0 <= x < x0+w and y0 <= y < y0+h:  # 메뉴 클릭 여부 검사
                if i < 6:                       # 그리기 명령이면
                    mouse_mode = 0                  # 마우스 상태 초기화
                    draw_mode = i                   # 그리기 모드
                else:                           # 일반 명령이면
                    command(i)
                return

        pt2 = (x, y)                                # 종료좌표 저장
        mouse_mode = 1                              # 버튼 떼기 상태 지정

    elif event == cv2.EVENT_LBUTTONDOWN:            # 왼쪽 버튼 누르기
        pt1 = (x, y)  # 시작좌표 저장
        mouse_mode = 2

    if mouse_mode >= 2:  # 왼쪽 버튼 누르기 또는 드래그
        mouse_mode = 0 if x < 125 else 3  # 메뉴 영역 확인- 마우스 상태 지정
        pt2 = (x, y)

def draw(image, color=(200, 200, 200)):
    global draw_mode, thickness, pt1, pt2

    if draw_mode == DRAW_RECTANGLE:                      # 사각형 그리기
        cv2.rectangle(image, pt1, pt2, color, thickness)
        
    elif draw_mode == DRAW_LINE:                         # 직선 그리기
        cv2.line(image, pt1, pt2, color, thickness)
        
    elif draw_mode == DRAW_BRUSH:                        # 브러시 그리기 
        cv2.line(image, pt1, pt2, color, thickness * 3)
        pt1 = pt2                                   # 종료 좌표를 시작 좌표로 지정
        
    elif draw_mode == ERASE:                             # 지우개
        cv2.line(image, pt1, pt2, (255, 255, 255), thickness * 5)
        pt1 = pt2

    elif draw_mode == DRAW_CIRCLE:                      # 원 그리기
        d = np.subtract(pt1, pt2)                       # 두 좌표 차분
        radius = int(np.sqrt(d[0] ** 2 + d[1] ** 2))
        cv2.circle(image, pt1, radius, color, thickness)

    elif draw_mode == DRAW_ECLIPSE:  # 타원 그리기
        center = np.abs(np.add(pt1, pt2)) // 2  # 두 좌표의 중심점 구하기
        size = np.abs(np.subtract(pt1, pt2)) // 2  # 두 좌표의 크기의 절반
        cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)

    cv2.imshow("PaintCV", image)

image = np.full((500, 800, 3), 255, np.uint8)
icons = place_icons(image, (60, 60))                # 아이콘 배치, 아이콘 크기
x, y, w, h = icons[-1]                               # 아이콘 사각형 마지막 원소
palatte_roi  = (0, y + h + 2  , 120, 120)           # 팔레트 ROI
hueIndex_roi = (0, y + h + 124, 120, 15)            # 색상인덱스 ROI

icons.append(palatte_roi)      # 팔레트 사각형 추가
icons.append(hueIndex_roi)  # 색상인덱스 사각형 추가
create_colorPlatte(image, 0, icons[PALETTE])    # 팔레트 생성
create_hueIndex(image, icons[HUE_IDX])                      # 색상인텍스 생성
cv2.imshow("PaintCV", image)
cv2.setMouseCallback("PaintCV", onMouse)                 # 마우스 콜백 함수

while True:
    if mouse_mode == 1:                                # 마우스 버튼 떼기
        draw(image, Color)                             # 원본에 그림
    elif mouse_mode == 3:                              # 마우스 드래그 
        if draw_mode == DRAW_BRUSH or draw_mode == ERASE:
            draw(image, Color)                         # 원본에 그림
        else:
            draw(np.copy(image), (200, 200, 200))      # 복사본에 회색으로 그림
   
    if cv2.waitKey(30) == 27:                          # ESC 키를 누르면 종료 
        break
