from coin_preprocess import *
from coin_utils import *                            # 기타 함수
from Common.utils import put_string

coin_no = int(input("동전 영상 번호: "))
image, th_img = preprocessing(coin_no)                              # 전처리 수행
circles = find_coins(th_img)                     # 객체(회전사각형) 검출
coin_imgs = make_coin_img(image, circles)                  # 동전 영상 생성
coin_hists= [calc_histo_hue(coin) for coin in coin_imgs]   # 동전 영상 히스토그램

groups = grouping(coin_hists)                              # 동전 영상 그룹 분리
ncoins = classify_coins(circles, groups)                   # 동전 인식

coin_value = np.array([10, 50, 100, 500])                             # 동전 금액
for i in range(4):
    print("%3d원: %3d개" % (coin_value[i], ncoins[i]))

total = sum(coin_value * ncoins )           # 동전금액* 동전별 개수
str = "Total coin: {:,} Won".format(total)            # 계산된 금액 문자열
print(str)                                                 # 콘솔창에 출력
put_string(image, str, (650, 50), '', (0,230,0))

## 동전 객체에 정보(반지름, 금액) 표시
color = [(0, 0, 250), (255, 255, 0), (0, 250, 0), (250, 0, 255)]  # 동전별 색상
for i, (c, r) in enumerate(circles):
    cv2.circle(image, c, r, color[groups[i]], 2)
    put_string(image, i, (c[0] - 15, c[1] - 10), '', color[2])  # 검출 순번과 동전 반지 표시
    put_string(image, r, (c[0], c[1] + 15), '', color[3])

cv2.imshow("result image", image)
key = cv2.waitKey(0)  # 키 이벤트 대기
