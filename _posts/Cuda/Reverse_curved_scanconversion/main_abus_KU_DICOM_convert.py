# 이 스크립트는 DICOM 형식의 의료 이미지 데이터를 처리하고 다양한 변환 및 재구성 작업을 수행하는 목적으로 작성되었습니다. 
# 데이터 처리 및 보간, 변환, 결과 데이터의 저장

#! usr/bin/env python

from functions import * # 사용자 정의 함수
import nrrd, pydicom, natsort # 데이터 파일 및 DICOM 파일을 처리
import matplotlib.pyplot as plt # 데이터 시각화
from scipy.interpolate import RegularGridInterpolator # 데이터 보간
from math import sin, cos # 삼각 함수
import time # 시간 측정
from collections import OrderedDict # 순서가 있는 딕셔너리를 사용


# 다양한 데이터 형식에 대한 매핑 정보
dtypeDict = {"char" : "str",
             "unsigned char" : "uint8",
             "short" : "int16",
             "unsigned short" : "uint16",
             "int" : "int32",
             "unsigned int" : "uint32",
             "long" : "uint32",
             "unsigned long" : "uint32",
             "long long" : "int64",
             "unsigned long long" : "uint64",
             "float" : "float32",
             "double" : "float64",
             "bool" : "bool"}

# 파일 경로 및 이름 목록을 가져오는 함수
def globFileList(fileNames, recursive=False, directoryOnly=False, isFullPath=True):
    """
    주어진 파일 경로 또는 파일 목록에 대해 일치하는 파일 목록을 검색하고 반환합니다.

    Args:
    fileNames (str 또는 list): 파일 경로 또는 파일 목록입니다. 단일 파일 경로 또는 파일 경로 목록을 지정할 수 있습니다.
    recursive (bool, optional): 하위 디렉터리를 검색하여 재귀적으로 파일을 찾을지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
    directoryOnly (bool, optional): 디렉터리만 반환할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
    isFullPath (bool, optional): 반환된 파일 경로가 전체 경로인지 파일 이름만 포함하는지 여부를 나타내는 부울 값입니다. 
                                기본값은 True입니다.

    Returns:
    list: 일치하는 파일 또는 디렉터리의 목록을 나타내는 문자열 목록을 반환합니다.

    Examples:
    # 현재 디렉터리에서 모든 파일 목록을 가져오기
    file_list = globFileList('*')

    # 특정 디렉터리에서 재귀적으로 모든 파일 목록 가져오기
    file_list = globFileList('/path/to/directory', recursive=True)

    # 디렉터리만 반환하고 전체 경로가 아닌 경우
    dir_list = globFileList('/path/to/directory', directoryOnly=True, isFullPath=False)
    """    
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

# DICOM 파일에서 특정 태그의 값을 가져오는 함수. description
def get_Tag_Val_By_Desc(dicomObj, targetDesc):
    for key in dicomObj.keys():
        desc = str(ds[key].description())
        if desc.lower() == targetDesc.lower():
            return str(ds[key].value)

# DICOM 파일에서 특정 태그의 값을 가져오는 함수, address
def get_Tag_Val_By_addr(dicomObj, targetAddr):
    for tag, key in zip(dicomObj, dicomObj.keys()):
        if tag.tag == targetAddr:
            return dicomObj[key].value

isDebug = False # 변수가 True인 경우 디버그 관련 정보와 이미지를 출력
targetPrecision = 0.05
targetResamplingUnitLength = 0.150 # (mm)


if __name__ == "__main__":
    # 선택한 DICOM 디렉토리의 파일 목록을 가져오고, 필요한 데이터를 처리하여 재구성
    # 데이터 재구성 및 처리에는 데이터 보간 및 다른 작업이 포함
    # 결과 데이터는 NRRD 형식으로 저장

    

    dcmFilePathNameSelected = select_dir_or_file(statement = "Select a target pair DICOM directory.", target="dir", isMuliple=False)
    dicomFileNameList = globFileList(f"{dcmFilePathNameSelected}/*.*", recursive=True, directoryOnly=False, isFullPath=True)
    dicomUSFileNameList = natsort.natsorted([a for a in dicomFileNameList if "\\US\\" in a ]) # DICOM 파일 목록에서 "\US\" 문자열이 포함된 파일 경로들을 자연 정렬(natural sort)하여 정렬
    priorDirNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[1] for a in dicomUSFileNameList] # 각 DICOM 파일 경로에 대해 추출한 2번째 디렉터리 이름 저장
    dataNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[2].replace("-", "") for a in dicomUSFileNameList] # 각 DICOM 파일 경로에 대해 추출한 데이터 이름이 저장
    # print(f'dicomUSFileNameList: {dicomUSFileNameList}')
    # print(f'priorDirNameList: {priorDirNameList}')
    # print(f'dataNameList: {dataNameList}')

    # get geometry information from DICOM
    for num, (dicomUSFileName, priorDirName, dataName) in enumerate(zip(dicomUSFileNameList, priorDirNameList, dataNameList)):
        prevtime = time.time() # 시작
        ds = pydicom.read_file(dicomUSFileName, stop_before_pixels=False) # DICOM 파일의 메타데이터(헤더-태그, 값, 속성)와 픽셀 데이터, 파일 포맷, 압축 및 인코딩 정보, 이미지 크기, 해상도, 촬영 조건, 환자 정보, 스캔 장비 정보 등
        dicomViewName = get_Tag_Val_By_Desc(ds, "Series Description") # RMED(Right Medial), RAP(Right Anterior Posterior), RLAT(Right Lateral)
        # print(f'dicomViewName: {dicomViewName}')

        # get DICOM tag
        PixelSpacing = get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0028', '0030')) # 가로 0.073mm, 세로 0.200mm
        # print(f'PixelSpacing: {PixelSpacing}') 

        rangeSpacing = float(PixelSpacing[0]) # 가로 0.073mm
        azimuthalSpacing = float(PixelSpacing[1]) # 세로 0.200mm
        sliceSpacing = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0018', '0088')))
        probeRadius = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1040')))
        maxCut = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1061')))
        metadata = {'PixelSpacing' : (rangeSpacing, azimuthalSpacing),
                    'SpacingBetweenSlices': sliceSpacing,
                    'CurvatureRadiusProbe' : probeRadius,
                    'MaxCut' : maxCut}
        # print(f'sliceSpacing: {sliceSpacing}, probeRadius: {probeRadius}, maxCut: {maxCut}')
        # sliceSpacing: 0.475674, probeRadius: 442.11, maxCut: 0.0566474
        # maxCut: steered compouning에서, footprint를 1로 할 때, 버려지는 영역의 삼각형 밑변의 길이. 측면의 휘어진 선이 포함되는 것을 잘라줌
        dicom_image = ds.pixel_array 
        # print(f'dicom_image.shape: {dicom_image.shape}') 
        # dicom_image.shape: (348, 682, 865)
        # slice x width x height
        # plt.figure
        # plt.imshow(dicom_image[0, :, :])
        # plt.show()
        dataName = f"{dataName.upper()}_{dicomViewName.upper()}" # dataName: ABUS003_RMED
        # print(f'dataName: {dataName}')
        # load dicomData
        dicomData = np.zeros([dicom_image.shape[2], dicom_image.shape[1], dicom_image.shape[0]]) # dicomData.shape: (865, 682, 348) <- (348, 682, 865)
        # print(f'dicomData.shape: {dicomData.shape}')

        # 각 슬라이스에 대한 마스크 적용
        # elaps time: 1.5 sec
        for s in range(dicomData.shape[2]): # dicomData.shape: (865, 682, 348), slice direction, 348
            tmpData = dicom_image[s, :, :].T # dicom_image: slice x width(range) x height -> dicomData: slice x height x width. dicom -> nrrd 저장순서
            # print(f'tmpData.shape: {tmpData.shape}') # (865, 682) 긴 방향이 azimuth
            # plt.figure()
            # plt.imshow(tmpData)
            # plt.show()
            if s == 0: # 
                # conduct stencil mask
                # 각 range 방향별로 몇 개를 clip할 지 미리 계산
                clip_size = np.round(np.linspace(np.round(tmpData.shape[0] * maxCut), 0, tmpData.shape[1]))
                # print(f'clip_size: {clip_size}')
                # print(f'tmpData.shape[0]: {tmpData.shape[0]}, * maxCut: {maxCut}, tmpData.shape[1]: {tmpData.shape[1]}')
                # tmpData.shape[0]: 865, * maxCut: 0.0566474, tmpData.shape[1]: 682
                # [49, 49, 49, ..., 48, 48, ..., 0, 0, 0]
                # height(=range), width number
                stencil_mask = np.ones((tmpData.shape[0], tmpData.shape[1]))
                # stencil_mask.shape = (865, 682)
                for idx, clip in zip(range(stencil_mask.shape[1]), clip_size): # 682 range 방향으로 하나씩. 각 azimuth에 대해서
                    # print(f'idx, clip: {idx}, {clip}')
                    # idx: 0-681 (width), clip: 49-0
                    if clip != 0:
                        stencil_mask[:int(clip), idx] = 0
                        stencil_mask[-int(clip):, idx] = 0
            dicomData[:, :, s] = tmpData * stencil_mask # 0th slice에서 생성된 스텐실 마스크 사용
            # plt.figure()
            # plt.imshow(tmpData)
            # plt.figure()
            # plt.imshow(tmpData * stencil_mask)
            # plt.show()

        


        # dicomHeader: OrderedDict([
        # ('type', 'uint8'),
        # ('dimension', 3), 
        # ('space', 'left-posterior-superior'), 
        # ('kinds', ['domain', 'domain', 'domain']), 
        # ('encoding', 'raw'), 
        # ('sizes', [1144, 382, 348]), 
        # ('space directions', array([[0.15, 0., 0.], [0., 0.15, 0.], [0., 0., 0.475674]])), 
        # ('space origin', array([-85.725, 0., -82.529439])), 
        # ('meta info', 
        #   {'PixelSpacing': (0.073, 0.2), 
        #   'SpacingBetweenSlices': 0.475674, 
        #   'CurvatureRadiusProbe': 442.11, 
        #   'MaxCut': 0.0566474})
        # ])

        ###
        dicomHeader = OrderedDict()
        dicomHeader["type"] = "unsigned char"
        dicomHeader["dimension"] = len(dicomData.shape)
        dicomHeader["space"] = "left-posterior-superior"
        dicomHeader["kinds"] = ["domain", "domain", "domain"]
        dicomHeader["encoding"] = "raw"
        dicomHeader["sizes"] = np.array(dicomData.shape)

        # set RA coordinate of src data : R = srcRangeA (mm), A = srcAngleA (degree)
        tmpData = dicomData[..., int(dicomHeader["sizes"][2]/2)] # center slice indexing
        # plt.figure
        # plt.imshow(tmpData)
        # plt.show()
        # 
        # print(f'dicomData.shape: {dicomData.shape}, int(dicomHeader["sizes"][2]/2: {int(dicomHeader["sizes"][2]/2)}')
        # dicomData.shape: (865, 682, 348), int(dicomHeader["sizes"][2]/2: 174
        # print(f'tmpData.shape[0]: {tmpData.shape[0]}') 
        center_IJK = [(tmpData.shape[0]-1) / 2.0, 0] # [865 / 2, 0] = [432, 0]. azimuth center index at surface
        srcRangeA = probeRadius - (np.arange(tmpData.shape[1]) - center_IJK[1]) * rangeSpacing # 인덱스 아니고 실제 좌표계 in mm
        # print(f'srcRangeA: {srcRangeA}, probeRadius: {probeRadius}, tmpData.shape[1]: {tmpData.shape[1]}, center_IJK[1]: {center_IJK[1]}, rangeSpacing: {rangeSpacing}')
        # 
        # srcRangeA (mm): [442.11-392.397]: range 방향으로 얼마나 떨어져 있는지
        # probeRadius: 442.11 (dicom tag), 
        # tmpData.shape[1]: 682 (range samples), 
        # center_IJK[1]: 0, (center_IJK는 표면의 center position인데, [1]은 range 0 즉 표면의 위치값)
        # rangeSpacing: 0.073 # 0.073mm, 픽셀간격
        srcAngleA = (np.arange(tmpData.shape[0]) - center_IJK[0]) * azimuthalSpacing / probeRadius * 180 / np.pi
        # print(f'srcAngleA: {srcAngleA}, tmpData.shape[0]: {tmpData.shape[0]}, center_IJK[0]: {center_IJK[0]}, azimuthalSpacing: {azimuthalSpacing}, probeRadius: {probeRadius}')
        # srcAngleA = -11.19711237 ~ 11.19711237 degrees (180/np.pi)
        # tmpData.shape[0]: 865,  인덱스
        # center_IJK[0]: 432.0, 인덱스
        # azimuthalSpacing: 0.2, 라디안
        # probeRadius: 442.11
        # 180/np.pi: radian -> degree
        # index -> 실제 좌표계로 변경
        
        ## srcRangeA, srcAngleA: reverse scan conversion하기 전의 실제 좌표

        # set Cartesian coordinate for destinated data
        # meanSampling = (azimuthalSpacing + rangeSpacing)/2.0 # Not used
        
        # targetResamplingUnitLength = 0.150 # (mm), 뭔가 비율이 안맞았던 것 같은데....rescaling
        targetXMin = (-1) * np.round(np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        #부채꼴의 좌측 최소값, x 위치를 구한 후 rescaling
        # a*sin(theta)
        # print(f'max(srcRangeA): {max(srcRangeA)}') # max(srcRangeA): 442.11
        # print(f'min(srcAngleA): {min(srcAngleA)}') # min(srcAngleA): -11.197112370066982
        # print(f'min(srcAngleA) / 180.0 * np.pi: {min(srcAngleA) / 180.0 * np.pi}') # -0.1954264775734546
        # print(f'sin(min(srcAngleA) / 180.0 * np.pi): {sin(min(srcAngleA) / 180.0 * np.pi)}') # -0.1941849121578608
        # print(f'np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)): {np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi))}') # 85.85109151411184
        # print(f'np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength): {np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength)}') # 572.0
        targetXMax = np.round(np.round(max(srcRangeA) * sin(max(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        # 부채꼴의 우측 최대값, x 위치를 구한 후 rescaling
        targetYMin = np.round(np.round((min(srcRangeA)) * cos(min(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        # a*cos(theta)
        targetYMax = np.round(np.round(max(srcRangeA) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        #
        # print(f'targetXMin: {targetXMin}, targetXMax: {targetXMin}, targetYMin: {targetYMin}, targetYMax: {targetYMax}')
        # targetXMin: -85.8, targetXMax: -85.8, targetYMin: 384.9, targetYMax: 442.05

        X = [targetXMin] # -85.8 
        while(True): # 항상
            # print(f'X[-1]: {X[-1]}, targetResamplingUnitLength: {targetResamplingUnitLength}, targetXMax: {targetXMax}')
            # targetXMin 부터 targetResamplingUnitLength 간격으로 targetXMax미만의 array를 생성함
            # X = np.arange(targetXMin, targetXMax, targetResamplingUnitLength)와 동일함.
            if X[-1] + targetResamplingUnitLength > targetXMax:
                break
            else:
               X.append(X[-1] + targetResamplingUnitLength)
        # print(f'X: {np.round(X,2)}')

        Y = [targetYMin]
        while(True):
            if Y[-1] + targetResamplingUnitLength > targetYMax:
                break
            else:
               Y.append(Y[-1] + targetResamplingUnitLength)

        Xi, Yi = np.meshgrid(X, Y, indexing='ij') # 그림참조, reversed scan conversion할 사각형. 범위 X min ~ X max, Y min ~ Y max
        dstRangeMesh = np.sqrt(Xi**2 + Yi**2)
        dstAngleMesh = np.arctan(Xi / Yi) / np.pi * 180.0 # 위치에 해당하는 Mesh를 구해놓은 것
        # print(f'dstRangeMesh: {dstRangeMesh}')

        if isDebug:
            print("")
            print(f"srcRangeA (mm) = min:{min(srcRangeA)}, max:{max(srcRangeA)}")
            print(f"srcAngleA (degree) = min:{min(srcAngleA)}, max:{max(srcAngleA)}")
            print(f"dstRangeMesh = min:{np.min(dstRangeMesh)}, max:{np.max(dstRangeMesh)}")
            print(f"dstAngleMesh = min:{np.min(dstAngleMesh)}, max:{np.max(dstAngleMesh)}")

        # srcRangeA (mm) = min:392.39700000000005, max:442.11
        # srcAngleA (degree) = min:-11.197112370066982, max:11.197112370066982
        # dstRangeMesh = min:384.9, max:450.2997251831188
        # dstAngleMesh = min:-12.566629820672725, max:12.545356250389412

        currenttime = time.time()
        print(f'time elapse before transform: {currenttime - prevtime}')


        prevtime_transform = time.time()
        # main
        transformed_nrrdData = np.zeros((len(X), len(Y), dicomData.shape[2]), dtype=dtypeDict[dicomHeader["type"]]) # 결과 데이터 담을 곳
        # print(f'dicomData.shape[2]: {dicomData.shape[2]}') # dicomData.shape[2]: 348 # slice
        # transformed_nrrdData_ref = np.zeros((len(X), len(Y), dicomData.shape[2]), dtype=dtypeDict[dicomHeader["type"]]) # 하는 게 없는데, 그냥 zeros
        # retransformed_nrrdData_ref = np.zeros((dicomData.shape[0], dicomData.shape[1], dicomData.shape[2]), dtype=dtypeDict[dicomHeader["type"]]) # 하는 거 없음
        # 데이터 사이즈만 동일한 zeros, 하나는 dicom, 하나는 nrrd. 사용되진 않음.

        newTime = []
        for s in range(dicomData.shape[2]): # slice
            # data
            tmpData = dicomData[..., s] # 각 slice 마다 처리
            # if isDebug and False:
            #     temp = np.ones_like(tmpData)
            #     temp[tmpData != 0] = 255
            #     tmpData = temp
            # linear interp
            st = time.time()
            # main
            interp_func = RegularGridInterpolator((srcAngleA, srcRangeA), tmpData, method='linear', bounds_error=False, fill_value=0)
            transformed = np.fliplr(interp_func((dstAngleMesh, dstRangeMesh)))

            transformed[np.isnan(transformed)] = 0 # 후처리
            transformed = transformed.astype(dtypeDict[dicomHeader["type"]]) # 타입변경
            transformed_nrrdData[..., s] = transformed # 결과 데이터
            newTime.append(time.time() - st) 

        # isDebug=True
        if isDebug:
            print(f"new={np.sum(newTime)}")
            plt.figure()
            plt.title("transformed_nrrdData") # result nrrd data, 중앙 슬라이스
            plt.imshow(transformed_nrrdData[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # 타입 변경된 transformed data
            plt.colorbar()
            # plt.figure()
            # plt.title("transformed_nrrdData_ref")
            # plt.imshow(transformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # zeros, nrrd data의 데이터 사이즈와 같음
            # plt.colorbar()
            # plt.figure()
            # plt.title("transformed_nrrdData_ref-transformed_nrrdData") # zeros - transformed nrrd....  왜 확인하지?
            # # print(transformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T)
            # plt.imshow(np.abs(transformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T- transformed_nrrdData[..., int(transformed_nrrdData.shape[2]/2)].T), cmap="gray", vmin=0, vmax=255)
            # plt.colorbar()
            # check recovery
            plt.figure()
            plt.title("dicomData") # original input
            plt.imshow(dicomData[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # 원본
            plt.colorbar()
            # plt.figure()
            # plt.title("retransformed_nrrdData_ref")
            # plt.imshow(retransformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # zeros
            # plt.figure()
            # plt.title("dicomData-retransformed_nrrdData_ref") # 0-이미지 abs... 무슨 의미?
            # plt.imshow(np.abs(dicomData[..., int(transformed_nrrdData.shape[2]/2)].T- retransformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T), cmap="gray", vmin=0, vmax=255)
            # plt.colorbar()
            plt.show()

        currenttime_transform = time.time()
        print(f'time elapse of transform: {currenttime_transform - prevtime_transform}')

        # update header, then save data
        dicomHeader["sizes"] = np.array(transformed_nrrdData.shape)
        dicomHeader["space directions"] = np.array([[targetResamplingUnitLength, 0., 0.], [0., targetResamplingUnitLength, 0.], [0., 0., sliceSpacing]])
        dicomHeader["space origin"] = np.array([dicomHeader["space directions"][0, 0] * (dicomHeader["sizes"][0] - 1) * 0.5 * (-1), 0.0, dicomHeader["space directions"][2, 2] * (dicomHeader["sizes"][2] - 1) * 0.5 * (-1)])
        dicomHeader["meta info"] = metadata
        os.makedirs(f"./transformed_data_mask_KU", exist_ok=True)
        os.makedirs(f"./transformed_data_mask_KU/{priorDirName}", exist_ok=True)
        nrrd.write(f"./transformed_data_mask_KU/{priorDirName}/{dataName}_transformed.nrrd", transformed_nrrdData, dicomHeader)
        print(f"    {dicomUSFileName} is processed ({num+1}/{len(dicomUSFileNameList)})...")

        # print(f'dicomHeader: {dicomHeader}')