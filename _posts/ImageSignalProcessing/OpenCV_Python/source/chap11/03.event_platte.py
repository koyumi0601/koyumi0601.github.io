from paint_init import *
from paint_utils import *

def onMouse(event, x, y, flags, param):  # 콜백 함수
    global pt1, pt2, mouse_mode, draw_mode

    if event == cv2.EVENT_LBUTTONUP:  # 왼쪽 버튼 떼기
        pt2 = (x, y)  # 종료좌표 저장
        mouse_mode = 1  # 버튼 떼기 상태 지정

        for i, (x0, y0, w, h) in enumerate(icons):  # 메뉴아이콘 사각형 조회
            if x0 <= x < x0+w and y0 <= y < y0+ h:  # 메뉴 클릭 여부 검사
                if i < 6:  # 그리기 명령이면
                    mouse_mode = 0  # 마우스 상태 초기화
                    draw_mode = i  # 그리기 모드
                else:  # 일반 명령이면
                    command(i)
                return  # 버그 수정

    elif event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누르기
        pt1 = (x, y)  # 시작좌표 저장
        mouse_mode = 2

    if mouse_mode >= 2:  # 왼쪽 버튼 누르기 또는 드래그
        mouse_mode = 0 if x < 125 else 3  # 메뉴 영역 확인- 마우스 상태 지정
        pt2 = (x, y)

def command(mode):
    global icons, image, Color, hue

    if mode == PALETTE:  # 색상팔레트 영역 클릭 시
        pixel = image[pt2[::-1]]  # 팔레트 클릭 좌표 화소값
        x, y, w, h = icons[COLOR]
        image[y:y + h - 1, x:x + w - 1] = pixel  # 색상 아이콘 영역에 pixel색 지정
        Color = tuple(map(int, pixel))
        # Color = tuple(pixel.astype('int'))

    elif mode == HUE_IDX:                                # 색상인텍스 클릭 시         
        create_colorPlatte(image, pt2[0], icons[PALETTE])

    cv2.imshow("PaintCV", image)

image = np.full((500, 800, 3), 255, np.uint8)
icons = place_icons(image, (60, 60))                # 아이콘 배치, 아이콘 크기
x, y, w, h = icons[-1]                               # 아이콘 사각형 마지막 원소

icons.append((0, y + h + 2  , 120, 120))      # 팔레트 사각형 추가
icons.append((0, y + h + 124, 120, 15) )  # 색상인덱스 사각형 추가
create_colorPlatte(image, 0, icons[PALETTE])    # 팔레트 생성
create_hueIndex(image, icons[HUE_IDX])                      # 색상인텍스 생성

cv2.imshow("PaintCV", image)
cv2.setMouseCallback("PaintCV", onMouse)                  # 마우스 콜백 함수
cv2.waitKey(0)
