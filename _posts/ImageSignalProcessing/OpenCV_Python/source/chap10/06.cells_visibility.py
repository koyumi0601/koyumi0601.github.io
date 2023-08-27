import cv2
import matplotlib.pyplot as plt

def get_cell(img, j, i, size):
    x, y = (j * size[0], i * size[1])  # 숫자칸 시작좌표
    return img[y:y + size[1], x:x + size[0]]

train_image = cv2.imread('images/train_numbers.png', cv2.IMREAD_GRAYSCALE)
if train_image is None: raise Exception("영상 파일 읽기 에러")
train_image = train_image[5:405, 6:806]                 # 상하좌우 여백 제거

size, K = (40, 40),  15                                 # 숫자 영상 크기
nclass, nsample = 10, 20                                # 그룹수, 그룹당 샘플수
cells =[get_cell(train_image, j, i, size) for i in range(nclass) for j in range(nsample)]
# cells = [np.hsplit(row, nsample) for row in np.vsplit(train_image,nclass)]
# cells = np.reshape(cells, (-1,40,40))

for i, cell in enumerate(cells):
    plt.subplot(10, 20, i+1), plt.axis('off'), plt.imshow(cell, cmap='gray')
plt.tight_layout(), plt.show()