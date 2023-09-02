import numpy as np, cv2

def preprocessing(no):  # 검출 전처리
    image = cv2.imread('images/woman/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    return image, gray

def calc_center(eye, face):
    x, y, _, _ = np.add(face, eye)                          # 두 좌표 더하기
    _, _, w, h = np.divide(eye, 2).astype(int)              # 정수형 변환
    return tuple(np.add((x,y), (w,h)))                      # 튜플로 반환

def calc_rotMat(center, pts):
    pt0, pt1 = pts
    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0         # 두 좌표 스왑
    dx, dy = np.subtract(pt1, pt0)                   # 두 좌표간 차분 계산

    angle = cv2.fastAtan2(dy, dx)                   # 차분으로 기울기 계산
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    return rot_mat

def correct_image(image, rot_mat, eyes_centers):
    size = (image.shape[::-1])[1:]                  # 행태와 크기는 역순
    corr_image = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)

    homo_coord = np.append(eyes_centers, np.ones((2, 1)), axis=1) # 동차좌표 변환
    # # 행렬 처리 방식
    # center = cv2.gemm(homo_coord, rot_mat.T, 1, None, 1)
    # corr_center= tuple(map(tuple, center.astype(int)))   # 리스트 --> 튜플 변환
    # 파이썬 리스트 방식
    corr_center = [tuple(map(int, np.dot(rot_mat, c))) for c in homo_coord]
    return corr_image, corr_center                 # 보정 결과 반환

def define_roi(pt, size):
    return (pt[0], pt[1], size[0], size[1])

def detect_object(center, face):
    gap = np.array(face[2:4])
    gap1 = (gap * (0.45, 0.65)).astype(int)       # 평행이동 거리
    gap2 = (gap * (0.20, 0.1)).astype(int)
    center = np.array(center)

    pt1 = center - gap1        # 좌상단 평행이동 - 머리 시작좌표
    pt2 = center + gap1             # 우하단 평행이동 - 머리 종료좌표
    hair = define_roi(pt1, pt2-pt1)         # 머리 영역

    size = (np.array(hair[2:4]) * (1, 0.4)).astype(int)
    hair1 = define_roi(pt1, size)             # 윗머리 영역
    hair2 = define_roi(pt2-size, size)             # 귀밑머리 영역

    # 입력 영역 계산
    lip_center = center + (0, int(gap[1]* 0.3))
    lip1 = lip_center - gap2    # 좌상단 평행이동
    lip2 = lip_center + gap2         # 우하단 평행이동
    lip = define_roi(lip1, lip2-lip1 )

    return [hair1, hair2, lip, hair ]