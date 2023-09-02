import numpy as np, cv2
import matplotlib.pyplot as plt

def make_coin_img(src, circles):
    coins = []
    for center, radius  in circles:
        r = radius * 3                      # 검출 동전 반지름의 3 배
        cen = (r // 2, r // 2)                                  # 마스크 중심
        mask = np.zeros((r, r, 3), np.uint8)                    # 마스크 행렬
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)

        # 동전 영상 가져오기
        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)                      # 마스킹 처리
        coins.append(coin)                                      # 동전 영상 저장
        # cv2.imshow("mask_" + str(center) , mask)                    # 마스크 영상 보기
    return coins

def calc_histo_hue(coin):
    hsv = cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)  # 컬러 공간 변환
    hsize, ranges = [32], [0, 180]         # 32개 막대, 화소값 0~180 범위
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)
    return hist.flatten()

def grouping(hists):
    ws = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
          4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]        # 가중치 지정

    sim = np.multiply(hists, ws)
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)

    groups = [1 if s > 1.2 else 0 for s in similaritys]

    # x = np.arange(len(ws))
    # plt.plot(x, ws, 'r'), plt.show(), plt.tight_layout()	 # 가중치 그래프 보기
    # for i, s in enumerate(similaritys):											# 그룹핑 결과 출력	#
    #     print("%d %5f %d" % (i, s, groups[i]))
    return groups

# 동전 인식 : 동전 종류 결정
def classify_coins(circles, groups):
    ncoins = [0] * 4
    g = np.full((2,70), -1, np.int)
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3

    for group, (_, radius) in zip(groups, circles):
        coin = g[group, radius]
        ncoins[coin] += 1

    return np.array(ncoins)