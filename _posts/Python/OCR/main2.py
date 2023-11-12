from PIL import Image, ImageOps
import numpy as np
import os, natsort, time
import pytesseract
import matplotlib.pyplot as plt
import cv2

max8bitsInt = 2**8 - 1
max16bitsInt = 2**16 - 1

def load_image_gray8bits(fileName):
    return (np.dot(plt.imread(fileName)[..., :3], [0.2989, 0.5870, 0.1140]) * 255.0).astype("uint8")

def globFileList(fileNames, recursive=False, directoryOnly=False, isFullPath=True):
    import glob
    if type(fileNames) == str:
        fileNames = [fileNames]
    actualDirNames = list()
    actualFileNames = list()
    for fileName in fileNames:
        tmpStrList = fileName.replace('/', '\\').split('\\')
        if len(tmpStrList) == 1: 
            actualDirNames.append('.')
            actualFileNames.append(tmpStrList[-1])
        else:
            actualDirNames.append('\\'.join(tmpStrList[0:-1]))
            actualFileNames.append(tmpStrList[-1])
    fileNameList = list()
    if not(recursive):
        for dirName, fileName in zip(actualDirNames, actualFileNames):
            fileNameList = fileNameList + glob.glob(dirName + '\\' + fileName)
    else:
        for dirName, fileName in zip(actualDirNames, actualFileNames):
            for root, _, _ in os.walk(dirName):
                root = root.replace('\\', '/') + '/'
                fileNameList += glob.glob(root + '/*' + fileName + '*')
    fileListCleaned = list()
    for fullPathFileName in fileNameList:
        tmpFullPathFileName = os.path.abspath(fullPathFileName)
        if isFullPath:
            tmpPathFileName = tmpFullPathFileName
        else:
            tmpPathFileName = os.path.split(tmpFullPathFileName)[-1]
        if directoryOnly:
            if os.path.isdir(tmpFullPathFileName):
                fileListCleaned.append(tmpPathFileName)
        else:
            fileListCleaned.append(tmpPathFileName)
    return fileListCleaned

def inference_single_digit(img):
    padOffset = 1
    if np.sum(img)  == 0:
        return "0"
    else:
        padImg = np.pad(img, [(padOffset, padOffset), (padOffset+2, padOffset+2)], mode='constant', constant_values=0)
        return pytesseract.image_to_string(padImg, lang='eng').replace(" ", "").replace("\n", "")

def improved_image_processing(img):
    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return processed_img

if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = "D:/Program Files/Tesseract-OCR/tesseract.exe"
    offset = 0
    croppingCordinate = (1733-offset, 271-offset, 1772+offset, 286+offset)

    brightnessThres = 80
    minBrightness = 15
    maxBrightness = 190
    recalcMaxBrightness = maxBrightness - minBrightness

    digitWidth = 11
    digitInterval = 6
    startpoint_decimal_p2 = 0
    startpoint_decimal_p1 = 12
    startpoint_dicimal_n1 = 29

    screenWidth = 4
    startpoint_screening = 24

    imageProcess_time = []
    inference_time = []
    imgPathNameList = natsort.natsorted(globFileList("./testset/*.png"))
    for imgPathName in imgPathNameList:
        # imgPathName = r"D:\01.works\Tesseract_OCR\testset\image_00001.png"
        # image load
        st = time.time()
        imgGray = load_image_gray8bits(imgPathName)
        orgCroppedImg = imgGray[croppingCordinate[1]: croppingCordinate[3],croppingCordinate[0]: croppingCordinate[2]].astype(int)
        normCroppedImg = orgCroppedImg - minBrightness
        if brightnessThres < np.mean(normCroppedImg):
            normCroppedImg = np.abs(normCroppedImg - recalcMaxBrightness)
        normCroppedImg = np.pad(normCroppedImg, [(0, 0), (0, 1)], mode='constant', constant_values=0).astype("uint8") # special pad for this system
        # screen decimal
        normCroppedImg[:, startpoint_screening:startpoint_screening+screenWidth] = 0
        # split image to digits
        img_decimal_p2 = normCroppedImg[:, startpoint_decimal_p2:startpoint_decimal_p2+digitWidth]
        img_decimal_p1 = normCroppedImg[:, startpoint_decimal_p1:startpoint_decimal_p1+digitWidth]
        img_decimal_n1 = normCroppedImg[:, startpoint_dicimal_n1:startpoint_dicimal_n1+digitWidth]
        imageProcess_time.append(time.time()-st)
        recon_digits = np.zeros((img_decimal_p1.shape[0], digitWidth * 3 + digitInterval * 2), dtype="uint8")
        recon_digits[:, 0:0+digitWidth] = img_decimal_p2
        recon_digits[:, digitWidth+digitInterval:digitWidth+digitInterval+digitWidth] = img_decimal_p1
        recon_digits[:, (digitWidth+digitInterval)*2:(digitWidth+digitInterval)*2+digitWidth] = img_decimal_n1


        # plt.figure()
        # plt.imshow(normCroppedImg, cmap="gray")
        # plt.figure()
        # plt.imshow(img_decimal_p2, cmap="gray")
        # plt.figure()
        # plt.imshow(img_decimal_p1, cmap="gray")
        # plt.figure()
        # plt.imshow(img_decimal_n1, cmap="gray")
        # plt.figure()
        # plt.imshow(recon_digits, cmap="gray")
        # plt.show()

        # inference
        st = time.time()
        try:
            # # inference each digits
            # str_decimal_p2 = inference_single_digit(img_decimal_p2)
            # str_decimal_p1 = inference_single_digit(img_decimal_p1)
            # str_decimal_n1 = inference_single_digit(img_decimal_n1)
            # float_digits_each = float(f"{str_decimal_p2}{str_decimal_p1}.{str_decimal_n1}")
            # print(f"{str_decimal_p2}{str_decimal_p1}.{str_decimal_n1}")

            # inference whole digits
            padOffset = 4
            padCroppedImg = np.pad(recon_digits, [(padOffset, padOffset), (padOffset, padOffset)], mode='constant', constant_values=0)



            padCroppedImg_improved = improved_image_processing(padCroppedImg)


            float_digits_whole = float(pytesseract.image_to_string(padCroppedImg_improved, lang='eng').replace(" ", "").replace("\n", ""))
            inferred_depth = float_digits_whole / 10.0
        except:
            print((pytesseract.image_to_string(padCroppedImg, lang='eng')))
            plt.figure()
            plt.imshow(padCroppedImg, cmap="gray")
            plt.show()
        inference_time.append(time.time()-st)

        # save results
        plt.figure()
        plt.imshow(orgCroppedImg, cmap="gray")
        plt.text(1, 2, f"{inferred_depth}", fontsize=12, bbox=dict(facecolor='white', alpha=1.0))
        os.makedirs(f"./resultant_images", exist_ok=True)
        plt.savefig(f"./resultant_images/{os.path.basename(imgPathName)}.png")
        plt.close("all")
        print(f"    {os.path.basename(imgPathName)} is processing... the minimun required image size is {padCroppedImg.shape}.")

    print(f"number of inferred frames is {len(imgPathNameList)} ea.")
    print(f"total ellipesed time : pre = {np.round(np.sum(inference_time), 3)} sec, inf = {np.round(np.sum(inference_time), 3)} sec")
    print(f"mean ellipesed time : pre = {np.round(np.mean(inference_time), 3)} sec, inf = {np.round(np.mean(inference_time), 3)} sec")
    print(f"max ellipesed time : pre = {np.round(np.max(inference_time), 3)} sec, inf = {np.round(np.max(inference_time), 3)} sec")
    print(f"min ellipesed time : pre = {np.round(np.min(inference_time), 3)} sec, inf = {np.round(np.min(inference_time), 3)} sec")
