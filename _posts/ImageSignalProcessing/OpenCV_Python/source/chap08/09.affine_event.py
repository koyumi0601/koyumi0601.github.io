import numpy as np, cv2

def contain_pts(p, p1, p2):                 # p가 2개 좌표 범위 내 검사
    return p1[0] <= p[0] < p2[0] and p1[1] <= p[1] < p2[1]

def draw_rect(title, img, pts):
    rois = [(p - small, small * 2) for p in pts]       # 12
    for (x,y), (w,h) in np.int32(rois):
        cv2.rectangle(img, (x,y,w,h), (0, 255, 0), 2)
    cv2.imshow(title, img)

def affine(img):
    aff_mat = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, aff_mat, image.shape[1::-1], cv2.INTER_LINEAR)
    draw_rect("image", np.copy(image), pts1)
    draw_rect('dst', dst, pts2)

def onMouse(event, x, y, flags, param):
    global check
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(pts1):
            p1, p2 = p - small, p + small
            if contain_pts((x, y), p1, p2): check = i

    if event == cv2.EVENT_LBUTTONUP: check = -1

    if check >= 0:
        pts1[check] = (x, y)
        affine(np.copy(image))

image = cv2.imread('images/affine1.jpg')
if image is None: raise Exception("영상파일 읽기 에러")

small = np.array((12, 12))
check = -1
pts1 = np.float32([(30, 30), (450, 30), (200, 370)])
pts2 = np.float32([(30, 30), (450, 30), (200, 370)])

draw_rect('image', np.copy(image), pts1)
draw_rect('dst', np.copy(image), pts2)
cv2.setMouseCallback("image", onMouse, 0)
cv2.waitKey(0)