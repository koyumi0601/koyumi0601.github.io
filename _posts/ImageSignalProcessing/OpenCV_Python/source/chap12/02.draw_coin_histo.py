from coin_preprocess import *
from coin_utils import *                            # 기타 함수
from Common.histogram import draw_histo_hue

coin_no = 15
image, th_img = preprocessing(coin_no)                            # 전처리 수행
circles = find_coins(th_img)                     # 객체(회전사각형) 검출
coin_imgs = make_coin_img(image, circles)                # 동전 영상 생성
coin_hists = [calc_histo_hue(coin) for coin in coin_imgs] # 영상 히스토그램

for i, img in enumerate(coin_imgs):
    h, w = 200, 256
    hist_img = draw_histo_hue(coin_hists[i], (h, w, 3))    # 색상 히스토그램 표시

    merge = np.zeros((h, w+h, 3), np.uint8)
    merge[:, :w] = hist_img
    merge[:, w:] = cv2.resize(img, (h, h))
    cv2.imshow("hist&coin-" + str(i), merge)

cv2.waitKey(0)
