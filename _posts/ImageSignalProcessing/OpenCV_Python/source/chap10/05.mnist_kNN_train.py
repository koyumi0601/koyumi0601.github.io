import cv2, numpy as np
import pickle, gzip, os
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

def load_mnist(filename):
    if not os.path.exists(filename):
        print("Downloading" )
        link = "http://figshare.com/ndownloader/files/25635053"
        urlretrieve(link, filename)
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def graph_image(data, lable, title, nsample):
    plt.figure(num=title, figsize=(6, 9))
    rand_idx = np.random.choice(range(data.shape[0]), nsample)
    for i, id in enumerate(rand_idx):
        img = data[id].reshape(28, 28)
        plt.subplot(6, 4, i + 1), plt.axis('off'), plt.imshow(img, cmap='gray')
        plt.title('%s: %d' % (title , lable[id]))
    plt.tight_layout()

train_set, valid_set, test_set = load_mnist('mnist.pkl.gz')
train_data, train_label = train_set
test_data, test_label = test_set
## MNIST 로드 데이터 크기 확인
print('train_set=', train_set[0].shape)
print('valid_set', valid_set[0].shape)
print('test_set', test_set[0].shape)

print('training...')
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)               # k-NN 학습 수행

nsample = 100
print("%d개 predicting..." % nsample)
_, resp, _ , _ = knn.findNearest(test_data[:nsample], k=5)            # k-NN 분류 수행
accur = sum(test_label[:nsample] == resp.flatten()) / len(resp)       # 성능 측정

print("정확도=", accur*100, '%')
graph_image(train_data, train_label, 'label', 24)                   # 학습 데이터 그리기
graph_image(test_data[:nsample], resp, 'predict', 24)                #
plt.show()