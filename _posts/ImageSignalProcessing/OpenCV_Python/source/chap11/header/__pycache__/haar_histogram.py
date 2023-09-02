import numpy as np, cv2
from Common.histogram import draw_histo


def draw_ellipse(image, sub, color, ratio, thickness=cv2.FILLED):
    x, y, w, h = sub
    center = (x + w // 2, y + h // 2)                   # 타원 중심
    size = (int(w * ratio), int(h * ratio))             # 타원 크기
    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)

def get_roi(image, rect):
    x, y, w, h = rect
    return image[y:y + h, x:x + w]

def make_masks(sub_roi, org_shape):                              # 영역별 마스크 생성
    base_mask = np.full(org_shape, 255, np.uint8)
    draw_ellipse(base_mask, sub_roi[3],  0, 0.45, -1)
    mask1 = get_roi(base_mask, sub_roi[0])  # 윗머리 마스크
    mask2 = get_roi(base_mask, sub_roi[1])  # 귀밑머리 마스크

    draw_ellipse(base_mask, sub_roi[2], 255, 0.40, -1)
    mask3 = get_roi(base_mask, sub_roi[2])
    mask4 = 255- get_roi(base_mask, sub_roi[3])

    return [mask1, mask2, mask3, mask4]

def calc_histo(image, sub_roi, masks):
    bins = (64, 64,64)  # 히스토그램 계급 개수
    ranges = (0,256, 0,256, 0,256)                                 # 각 채널 빈도 범위

    subs = [image[y:y + h, x:x + w] for x, y, w, h in sub_roi]
    hists = [cv2.calcHist([sub], [0,1,2], mask, bins, ranges) for sub, mask in zip(subs, masks)]
    hists = [ h / np.sum(h) for h in hists]

    hsv = [cv2.cvtColor(sub, cv2.COLOR_BGR2HSV) for sub in subs]
    hue_hist1 = cv2.calcHist([hsv[2]], [0], None, [90], [0, 180])  # Hue 채널 히스토그램 계산
    hue_hist2 = cv2.calcHist([hsv[3]], [0], None, [90], [0, 180])  # Hue 채널 히스토그램 계산

    # hist_img = draw_histo(hists[2][:,:, 2])
    # cv2.imshow("h", hist_img)

    # for i, mask in enumerate(masks):
    #     cv2.imshow("mask[" + str(i) + "]", masks[i])

    return  [hists[0], hists[1], hue_hist1, hue_hist2]