import numpy as np, cv2
from Common.fft2d import FFT, IFFT, calc_spectrum

def onRemoveMoire(val):
    radius = cv2.getTrackbarPos("radius", title)
    th  = cv2.getTrackbarPos("threshold", title)
    
    mask= cv2.threshold(spectrum_img, th, 255, cv2.THRESH_BINARY_INV)[1]
    y, x = np.divmod(mask.shape,2)[0]
    cv2.circle(mask, (x, y), radius, 255, -1)

    if dft.ndim<3:
        remv_dft = np.zeros(dft.shape, np.complex)
        remv_dft.imag = cv2.copyTo(dft.imag, mask=mask)
        remv_dft.real = cv2.copyTo(dft.real, mask=mask)
    else:
        remv_dft = cv2.copyTo(dft, mask=mask)

    result[:, image.shape[1]:] = IFFT(remv_dft, image.shape, mode)
    cv2.imshow(title, calc_spectrum(remv_dft))
    cv2.imshow("result", result)

image = cv2.imread('images/mo2.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 파일 읽기 에러")

mode = 3
result = cv2.repeat(image, 1, 2)                    # 원본 영상 + 결과 영상
dft, spectrum_img = FFT(image, mode)              # OpenCV dft() 함수 수행

title = "removed moire"
cv2.imshow("result", result)
cv2.imshow(title, spectrum_img)
cv2.createTrackbar("radius", title, 10, 255, onRemoveMoire)
cv2.createTrackbar("threshold", title, 120, 255, onRemoveMoire)
cv2.waitKey(0)