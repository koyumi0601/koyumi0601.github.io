import numpy as np, cv2
from Common.fft2d import fft2, ifft2, calc_spectrum, fftshift

def FFT(image, mode = 2):
    if mode == 1: dft = fft2(image)
    elif mode==2: dft = np.fft.fft2(image)
    elif mode==3: dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)                              # 셔플링
    spectrum = calc_spectrum(dft)               # 주파수 스펙트럼 영상
    return dft, spectrum

def IFFT(dft, shape, mode=2):
    dft = fftshift(dft)                                 # 역 셔플링
    if mode == 1: img = ifft2(dft).real
    if mode == 2: img = np.fft.ifft2(dft).real
    if mode ==3:  img = cv2.idft(dft, flags= cv2.DFT_SCALE)[:,:,0]
    img = img[:shape[0], :shape[1]]                 # 영삽입 부분 제거
    return cv2.convertScaleAbs(img)

image = cv2.imread('images/dft_240.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 파일 읽기 에러")
cy, cx = np.divmod(image.shape, 2)[0]                 # 행렬 중심점 구하기
mode = 3

dft, spectrum = FFT(image, mode)                  # FFT 수행 및 셔플링
lowpass = np.zeros(dft.shape, np.float32)
highpass = np.ones(dft.shape, np.float32)
cv2.circle(lowpass , (cx, cy), 30, (1,1), -1)
cv2.circle(highpass, (cx, cy), 30, (0,0), -1)

lowpassed_dft = dft * lowpass
highpassed_dft = dft * highpass
lowpassed_img = IFFT(lowpassed_dft, image.shape, mode)
highpased_img = IFFT(highpassed_dft, image.shape, mode)

cv2.imshow("image", image)
cv2.imshow("lowpassed_img", lowpassed_img) # 역푸리에 변환 영상
cv2.imshow("highpased_img", highpased_img)
cv2.imshow("spectrum_img", spectrum)
cv2.imshow("lowpass_spect", calc_spectrum(lowpassed_dft))
cv2.imshow("highpass_spect", calc_spectrum(highpassed_dft))
cv2.waitKey(0)