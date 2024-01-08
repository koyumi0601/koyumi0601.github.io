from functions import * # 사용자 정의 함수
import nrrd, pydicom, natsort # 데이터 파일 및 DICOM 파일을 처리
import matplotlib.pyplot as plt # 데이터 시각화
from scipy.interpolate import RegularGridInterpolator # 데이터 보간
from math import sin, cos # 삼각 함수
import time # 시간 측정
from collections import OrderedDict # 순서가 있는 딕셔너리를 사용
import glob

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

def get_all_files_in_directory_with_metadata(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if '/US/' in file_path:
                file_list.append(file_path)
    sorted_file_list = natsort.natsorted(file_list)
    sorted_file_list_wo_directory = [s.replace(directory, '').replace('-','') for s in sorted_file_list]
    prior_dir_name_list = [path.split('/')[1] for path in sorted_file_list_wo_directory]
    data_name_list = [path.split('/')[2] for path in sorted_file_list_wo_directory]
    return sorted_file_list, prior_dir_name_list, data_name_list

def get_Tag_Val_By_Desc(dicomObj, targetDesc):
    for key in dicomObj.keys():
        desc = str(ds[key].description())
        if desc.lower() == targetDesc.lower():
            return str(ds[key].value)

def get_Tag_Val_By_addr(dicomObj, targetAddr):
    for tag, key in zip(dicomObj, dicomObj.keys()):
        if tag.tag == targetAddr:
            return dicomObj[key].value

isDebug = False 
targetPrecision = 0.05
targetResamplingUnitLength = 0.150 # (mm)


if __name__ == "__main__":

    dcmFilePathNameSelected = select_dir_or_file(statement="Select a target DICOM directory.")
    dicomUSFileNameList, priorDirNameList, dataNameList = get_all_files_in_directory_with_metadata(dcmFilePathNameSelected)

    # get geometry information from DICOM
    for num, (dicomUSFileName, priorDirName, dataName) in enumerate(zip(dicomUSFileNameList, priorDirNameList, dataNameList)):
        prev_time_total = time.time()
        prev_time_dataload = time.time() # 시작
        ds = pydicom.read_file(dicomUSFileName, stop_before_pixels=False) # DICOM 파일의 메타데이터(헤더-태그, 값, 속성)와 픽셀 데이터, 파일 포맷, 압축 및 인코딩 정보, 이미지 크기, 해상도, 촬영 조건, 환자 정보, 스캔 장비 정보 등
        dicomViewName = get_Tag_Val_By_Desc(ds, "Series Description") # RMED(Right Medial), RAP(Right Anterior Posterior), RLAT(Right Lateral)

        # get DICOM tag
        PixelSpacing = get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0028', '0030')) # 가로 0.073mm, 세로 0.200mm
        rangeSpacing = float(PixelSpacing[0]) # 가로 0.073mm
        azimuthalSpacing = float(PixelSpacing[1]) # 세로 0.200mm
        sliceSpacing = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0018', '0088')))
        probeRadius = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1040')))
        maxCut = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1061')))
        metadata = {'PixelSpacing' : (rangeSpacing, azimuthalSpacing),
                    'SpacingBetweenSlices': sliceSpacing,
                    'CurvatureRadiusProbe' : probeRadius,
                    'MaxCut' : maxCut}
        dicom_image = ds.pixel_array 
        dataName = f"{dataName.upper()}_{dicomViewName.upper()}" # dataName: ABUS003_RMED

        # load dicomData
        dicomData = np.zeros([dicom_image.shape[2], dicom_image.shape[1], dicom_image.shape[0]]) # dicomData.shape: (865, 682, 348) <- (348, 682, 865)

        for s in range(dicomData.shape[2]): # dicomData.shape: (865, 682, 348), slice direction, 348
            tmpData = dicom_image[s, :, :].T # dicom_image: slice x width(range) x height -> dicomData: slice x height x width. dicom -> nrrd 저장순서
            # print(f'tmpData.shape: {tmpData.shape}') # (865, 682) 긴 방향이 azimuth

            if s == 0: # 
                clip_size = np.round(np.linspace(np.round(tmpData.shape[0] * maxCut), 0, tmpData.shape[1]))
                stencil_mask = np.ones((tmpData.shape[0], tmpData.shape[1]))
                for idx, clip in zip(range(stencil_mask.shape[1]), clip_size): # 682 range 방향으로 하나씩. 각 azimuth에 대해서
                    if clip != 0:
                        stencil_mask[:int(clip), idx] = 0
                        stencil_mask[-int(clip):, idx] = 0
            dicomData[:, :, s] = tmpData * stencil_mask # 0th slice에서 생성된 스텐실 마스크 사용        

        dicomHeader = OrderedDict()
        dicomHeader["type"] = "unsigned char"
        dicomHeader["dimension"] = len(dicomData.shape)
        dicomHeader["space"] = "left-posterior-superior"
        dicomHeader["kinds"] = ["domain", "domain", "domain"]
        dicomHeader["encoding"] = "raw"
        dicomHeader["sizes"] = np.array(dicomData.shape)

        # set RA coordinate of src data : R = srcRangeA (mm), A = srcAngleA (degree)
        tmpData = dicomData[..., int(dicomHeader["sizes"][2]/2)] # center slice indexing
        center_IJK = [(tmpData.shape[0]-1) / 2.0, 0] # [865 / 2, 0] = [432, 0]. azimuth center index at surface
        srcRangeA = probeRadius - (np.arange(tmpData.shape[1]) - center_IJK[1]) * rangeSpacing # 인덱스 아니고 실제 좌표계 in mm
        srcAngleA = (np.arange(tmpData.shape[0]) - center_IJK[0]) * azimuthalSpacing / probeRadius * 180 / np.pi
        targetXMin = (-1) * np.round(np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        targetXMax = np.round(np.round(max(srcRangeA) * sin(max(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        targetYMin = np.round(np.round((min(srcRangeA)) * cos(min(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
        targetYMax = np.round(np.round(max(srcRangeA) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)

        X = np.arange(targetXMin, targetXMax + targetResamplingUnitLength, targetResamplingUnitLength)
        Y = np.arange(targetYMin, targetYMax + targetResamplingUnitLength, targetResamplingUnitLength)
        Xi, Yi = np.meshgrid(X, Y, indexing='ij') # 그림참조, reversed scan conversion할 사각형. 범위 X min ~ X max, Y min ~ Y max
        dstRangeMesh = np.sqrt(Xi**2 + Yi**2)#.astype(np.float32)  # 원래 float64 였음. 3d 구성시 메모리 부족
        dstAngleMesh = (np.arctan(Xi / Yi) / np.pi * 180.0)#.astype(np.float32)  # 위치에 해당하는 Mesh를 구해놓은 것

        if isDebug:
            print("")
            print(f"srcRangeA (mm) = min:{min(srcRangeA)}, max:{max(srcRangeA)}")
            print(f"srcAngleA (degree) = min:{min(srcAngleA)}, max:{max(srcAngleA)}")
            print(f"dstRangeMesh = min:{np.min(dstRangeMesh)}, max:{np.max(dstRangeMesh)}")
            print(f"dstAngleMesh = min:{np.min(dstAngleMesh)}, max:{np.max(dstAngleMesh)}")


        # main

        # # case 1. 3d interpolation 1장씩 전체 slice에 넣기. 57초. 메모리 이슈 없음.
        # prev_time = time.time()
        # sliceIds = np.arange(dicomData.shape[2])
        # interp_func_3d = RegularGridInterpolator((srcAngleA, srcRangeA, sliceIds), dicomData, method='linear', bounds_error=False, fill_value=0)
        # transformed_3d_nrrdData = np.zeros((len(X), len(Y), dicomData.shape[2]), dtype=dtypeDict[dicomHeader["type"]]) # 결과 데이터 담을 곳
        # for s in range(dicomData.shape[2]): # slice
        #     dstZ_3d = np.ones(dstAngleMesh[:, :, np.newaxis].shape)*np.float64(s)
        #     tmp = interp_func_3d((dstAngleMesh[:, :, np.newaxis], dstRangeMesh[:, :, np.newaxis], dstZ_3d)) # 0.1770477294921875
        #     # 57 sec
        # print(f'check 3d interp time: {time.time()-prev_time}')


        # # case 2. 3d interpolation 32 장 정도 까지만 메모리 가능. 32장에 4초 이상 걸림
        # prev_time = time.time()
        # sliceIds = np.arange(dicomData.shape[2])
        # interp_func_3d = RegularGridInterpolator((srcAngleA, srcRangeA, sliceIds), dicomData, method='linear', bounds_error=False, fill_value=0)
        # z = np.arange(32) 
        # # 원래는 384 slice 
        # dstAngleMesh_3d_part = np.repeat(dstAngleMesh[:, :, np.newaxis], len(z), axis=2)
        # dstRangeMesh_3d_part = np.repeat(dstRangeMesh[:, :, np.newaxis], len(z), axis=2)
        # zg2d = np.repeat(z[:,np.newaxis], len(X), axis=1)
        # zg3d = np.repeat(zg2d[:,:,np.newaxis], len(Y), axis=2)
        # zg3d_t = zg3d.transpose(1, 2, 0)
        # print(dstAngleMesh_3d_part.shape)
        # print(zg3d_t.shape)
        # print(zg3d_t[:,:,1])
        # transformed_3d_nrrdData_part =  interp_func_3d((dstAngleMesh_3d_part, dstRangeMesh_3d_part, zg3d_t))
        # print(transformed_3d_nrrdData_part.shape)
        # print(f'partially processed: {time.time()-prev_time}')

        # case 3. 2d interpolation
            
        prev_time = time.time()
        transformed_nrrdData = np.zeros((len(X), len(Y), dicomData.shape[2]), dtype=dtypeDict[dicomHeader["type"]]) # 결과 데이터 담을 곳
        for s in range(dicomData.shape[2]): # slice
            tmpData = dicomData[..., s]
            interp_func = RegularGridInterpolator((srcAngleA, srcRangeA), tmpData, method='linear', bounds_error=False, fill_value=0)
            transformed = interp_func((dstAngleMesh, dstRangeMesh))
            transformed_nrrdData[..., s] = transformed 
            # 15.04
        print(f'org time for 1 image: {time.time()-prev_time}')

        transformed_nrrdData = np.fliplr(transformed_nrrdData)
        transformed_nrrdData[np.isnan(transformed_nrrdData)] = 0 # 후처리
        transformed_nrrdData = transformed_nrrdData.astype(dtypeDict[dicomHeader["type"]]) # 타입변경
        
        # isDebug=True
        if isDebug:
            plt.figure()
            plt.title("transformed_nrrdData") # result nrrd data, 중앙 슬라이스
            plt.imshow(transformed_nrrdData[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # 타입 변경된 transformed data
            plt.colorbar()
            plt.figure()
            plt.title("dicomData") # original input
            plt.imshow(dicomData[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255) # 원본
            plt.colorbar()
            plt.show()

        # update header, then save data
        dicomHeader["sizes"] = np.array(transformed_nrrdData.shape)
        dicomHeader["space directions"] = np.array([[targetResamplingUnitLength, 0., 0.], [0., targetResamplingUnitLength, 0.], [0., 0., sliceSpacing]])
        dicomHeader["space origin"] = np.array([dicomHeader["space directions"][0, 0] * (dicomHeader["sizes"][0] - 1) * 0.5 * (-1), 0.0, dicomHeader["space directions"][2, 2] * (dicomHeader["sizes"][2] - 1) * 0.5 * (-1)])
        dicomHeader["meta info"] = metadata
        os.makedirs(f"./transformed_data_mask_KU", exist_ok=True)
        os.makedirs(f"./transformed_data_mask_KU/{priorDirName}", exist_ok=True)
        nrrd.write(f"./transformed_data_mask_KU/{priorDirName}/{dataName}_transformed.nrrd", transformed_nrrdData, dicomHeader)
        print(f"    {dicomUSFileName} is processed ({num+1}/{len(dicomUSFileNameList)})...")
