#! usr/bin/env python

from functions import *
import nrrd, pydicom, natsort
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos
import time
from collections import OrderedDict

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

def get_Tag_Val_By_Desc(dicomObj, targetDesc):
    for key in dicomObj.keys():
        desc = str(ds[key].description())
        if desc.lower() == targetDesc.lower():
            return str(ds[key].value)

def get_Tag_Val_By_addr(dicomObj, targetAddr):
    for tag, key in zip(dicomObj, dicomObj.keys()):
        if tag.tag == targetAddr:
            return dicomObj[key].value

isSkipStencil = True
isDebug = False
targetPrecision = 0.05
doOldSC = False
doLabelUpdateAlso = False

if __name__ == "__main__":
    nrrdFilePathNameList = natsort.natsorted(select_dir_or_file(statement = "Select a target nrrd data file.", target="file", select_filter="nrrd File (*.nrrd)", isMuliple=True))
    labelFilePathNameList = natsort.natsorted(select_dir_or_file(statement = "Select a target label data file.", target="file", select_filter="label File (*.nrrd)", isMuliple=True))
    # dcmFilePathNameList = select_dir_or_file(statement = "Select a target pair DICOM directory.", target="dir", isMuliple=False)
    dcmFilePathNameList = r"D:\data_of_nrrds\ABUS_KU\Acquisition\DICOM\malignant"
    for nrrdFilePathName, labelFilePathName in zip(nrrdFilePathNameList, labelFilePathNameList):
        if os.path.basename(nrrdFilePathName).lower().replace(".nrrd", ".seg.nrrd") == os.path.basename(labelFilePathName).lower():
            nrrdDataFileName = os.path.split(nrrdFilePathName)[1]
            nrrdDataViewName = os.path.splitext(nrrdDataFileName)[0].split("_")[-1]
            dicomDirPathName = f"{dcmFilePathNameList}/ABUS-{'%03d' %(int(nrrdDataFileName[4:6]))}"
            dicomFileNameList = globFileList(f"{dicomDirPathName}/*.*", recursive=True, directoryOnly=False, isFullPath=True)
            dicomFileOnlyNameList = [a for a in dicomFileNameList if "\\US\\" in a ]
            # get geometry information from DICOM
            for dicomFileOnlyName in dicomFileOnlyNameList:
                ds = pydicom.read_file(dicomFileOnlyName, stop_before_pixels=False)
                dicomViewName = get_Tag_Val_By_Desc(ds, "Series Description")
                if dicomViewName.lower() == nrrdDataViewName.lower():
                    PixelSpacing = get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0028', '0030'))
                    rangeSpacing = float(PixelSpacing[0])
                    azimuthalSpacing = float(PixelSpacing[1])
                    sliceSpacing = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0018', '0088')))
                    probeRadius = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1040')))
                    maxCut = float(get_Tag_Val_By_addr(ds, pydicom.tag.Tag('0021', '1061')))
                    metadata = {'PixelSpacing' : (rangeSpacing, azimuthalSpacing),
                                'SpacingBetweenSlices': sliceSpacing,
                                'CurvatureRadiusProbe' : probeRadius,
                                'MaxCut' : maxCut}
                    dicom_image = ds.pixel_array
                    break
            
            # load dicomData
            dicomData = np.zeros([dicom_image.shape[2], dicom_image.shape[1], dicom_image.shape[0]])
            for s in range(dicomData.shape[2]):
                tmpData = dicom_image[s, :, :].T
                if s == 0:
                    # conduct stencil mask
                    clip_size = np.round(np.linspace(np.round(tmpData.shape[0] * maxCut), 0, tmpData.shape[1]))
                    stencil_mask = np.ones((tmpData.shape[0], tmpData.shape[1]))
                    for idx, clip in zip(range(stencil_mask.shape[1]), clip_size):
                        if clip != 0:
                            stencil_mask[:int(clip), idx] = 0
                            stencil_mask[-int(clip):, idx] = 0
                dicomData[:, :, s] = tmpData * stencil_mask

            # load nrrdData
            nrrdData, nrrdHeader= nrrd.read(nrrdFilePathName)
            labelData, labelHeader= nrrd.read(labelFilePathName)

            print("")
            for key, val in nrrdHeader.items():
                print(f"{key}: {val}, {type(val)}")
            print("")

            dicomHeader = OrderedDict()
            dicomHeader["type"] = "unsigned char"
            dicomHeader["dimension"] = len(dicomData.shape)
            dicomHeader["space"] = "left-posterior-superior"
            dicomHeader["kinds"] = ["domain", "domain", "domain"]
            dicomHeader["encoding"] = "raw"
            dicomHeader["sizes"] = np.array(dicomData.shape)

            for key, val in dicomHeader.items():
                print(f"{key}: {val}, {type(val)}")

            asdfasdf

            # conduct stencil mask
            tmpData = nrrdData[..., int(nrrdHeader["sizes"][2]/2)]
            if not(isSkipStencil):
                clip_size = np.round(np.linspace(np.round(tmpData.shape[0] * maxCut), 0, tmpData.shape[1]))
                stencil_mask = np.ones((tmpData.shape[0], tmpData.shape[1]))
                for idx, clip in zip(range(stencil_mask.shape[1]), clip_size):
                    if clip != 0:
                        stencil_mask[:int(clip), idx] = 0
                        stencil_mask[-int(clip):, idx] = 0
                stenciled_nrrdData = np.zeros_like(nrrdData)
                for s in range(stenciled_nrrdData.shape[2]):
                    stenciled_nrrdData[:, :, s] = nrrdData[:, :, s] * stencil_mask
            else:
                stenciled_nrrdData = nrrdData

            # set RA coordinate of src data : R = srcRangeA (mm), A = srcAngleA (degree)
            tmpData = nrrdData[..., int(nrrdHeader["sizes"][2]/2)]
            center_IJK = [(tmpData.shape[0]-1) / 2.0, 0]
            srcRangeA = probeRadius - (np.arange(tmpData.shape[1]) - center_IJK[1]) * rangeSpacing
            srcAngleA = (np.arange(tmpData.shape[0]) - center_IJK[0]) * azimuthalSpacing / probeRadius * 180 / np.pi

            # set Cart coordinate for destinated data
            meanSampling = (azimuthalSpacing + rangeSpacing)/2.0
            targetResamplingUnitLength = np.round(np.round(meanSampling / targetPrecision) * targetPrecision, 2)
            targetXMin = (-1) * np.round(np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetXMax = np.round(np.round(max(srcRangeA) * sin(max(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetYMin = np.round(np.round((min(srcRangeA)) * cos(min(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetYMax = np.round(np.round(max(srcRangeA) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            # X = np.linspace(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi), max(srcRangeA) * sin(max(srcAngleA) / 180.0 * np.pi), labelData.shape[0], dtype=float)
            # Y = np.linspace((min(srcRangeA)) * cos(min(srcAngleA) / 180.0 * np.pi), max(srcRangeA), labelData.shape[1], dtype=float)
            X = np.linspace(targetXMin, targetXMax, int((targetXMax - targetXMin)/ targetResamplingUnitLength), dtype=float)
            Y = np.linspace(targetYMin, targetYMax, int((targetYMax - targetYMin)/ targetResamplingUnitLength), dtype=float)
            Xi, Yi = np.meshgrid(X, Y, indexing='ij')
            dstRangeMesh = np.sqrt(Xi**2 + Yi**2)
            dstAngleMesh = np.arctan(Xi / Yi) / np.pi * 180.0

            if isDebug:
                print("")
                print(f"srcRangeA (mm) = min:{min(srcRangeA)}, max:{max(srcRangeA)}")
                print(f"srcAngleA (degree) = min:{min(srcAngleA)}, max:{max(srcAngleA)}")
                print(f"dstRangeMesh = min:{np.min(dstRangeMesh)}, max:{np.max(dstRangeMesh)}")
                print(f"dstAngleMesh = min:{np.min(dstAngleMesh)}, max:{np.max(dstAngleMesh)}")

            if doOldSC:
                # get Transform
                targetPoints_RAS_X = np.zeros((tmpData.shape[0], tmpData.shape[1]))
                targetPoints_RAS_Y = np.zeros((tmpData.shape[0], tmpData.shape[1]))
                angleRadList = []
                for j in range(tmpData.shape[1]):
                    for i in range(tmpData.shape[0]):
                        radius = probeRadius - (j - center_IJK[1]) * rangeSpacing # ung's note: radius = Roc - (range)
                        angleRad = (i - center_IJK[0]) * azimuthalSpacing / probeRadius # ung's note: theta = l/r = arc/radius
                        angleRadList.append(angleRad)
                        targetPoints_RAS_X[i, j] = -radius * sin(angleRad)
                        targetPoints_RAS_Y[i, j] = radius * cos(angleRad) - probeRadius
                meanSampling = (azimuthalSpacing + rangeSpacing)/2.0
                targetResamplingUnitLength = np.round(np.round(meanSampling / targetPrecision) * targetPrecision, 2)
                targetXMin = (-1) * np.round(np.round(np.abs(np.min(targetPoints_RAS_X)) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
                targetXMax = np.round(np.round(np.max(targetPoints_RAS_X) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
                targetYMin = np.round(np.round(np.min(targetPoints_RAS_Y) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
                targetYMax = np.round(np.round(np.max(targetPoints_RAS_Y) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
                XX = np.linspace(targetXMax, targetXMin, labelData.shape[0], dtype=float)
                YY = np.linspace(targetYMax, targetYMin, labelData.shape[1], dtype=float)
                XXi, YYi = np.meshgrid(XX, YY, indexing='ij')
                tri = Delaunay(np.column_stack([targetPoints_RAS_X.flatten(), targetPoints_RAS_Y.flatten()]))

            # main
            transformed_nrrdData = np.zeros((len(X), len(Y), stenciled_nrrdData.shape[2]), dtype=dtypeDict[nrrdHeader["type"]])
            transformed_nrrdData_ref = np.zeros((len(X), len(Y), stenciled_nrrdData.shape[2]), dtype=dtypeDict[nrrdHeader["type"]])
            newTime = []
            oldTime = []
            for s in range(stenciled_nrrdData.shape[2]):
                # data
                tmpData = stenciled_nrrdData[..., s]
                if isDebug:
                    temp = np.ones_like(tmpData)
                    temp[tmpData != 0] = 255
                    tmpData = temp
                # linear interp
                st = time.time()
                interp_func = RegularGridInterpolator((srcAngleA, srcRangeA), tmpData, method='linear', bounds_error=False, fill_value=0)
                transformed = np.fliplr(interp_func((dstAngleMesh, dstRangeMesh)))
                transformed[np.isnan(transformed)] = 0
                transformed = transformed.astype(dtypeDict[nrrdHeader["type"]])
                transformed_nrrdData[..., s] = transformed
                newTime.append(time.time() - st)
                if doOldSC:
                    # Delanuay linear interp
                    st = time.time()
                    interp_func_ref = LinearNDInterpolator(tri, tmpData.flatten())
                    transformed_ref = np.squeeze(interp_func_ref(XXi, YYi))
                    transformed_ref[np.isnan(transformed_ref)] = 0
                    transformed_ref = transformed_ref.astype(dtypeDict[nrrdHeader["type"]])
                    transformed_nrrdData_ref[..., s] = transformed_ref
                    oldTime.append(time.time() - st)
                print(f"    {nrrdFilePathName} is processed ({int(s)+1}/{int(stenciled_nrrdData.shape[2])})...")
                if doLabelUpdateAlso:
                    if transformed_nrrdData.shape != labelData.shape:
                        newLabelData = np.zeros_like(transformed_nrrdData)
                        for k in range(newLabelData.shape[2]):
                            print("")
                            for key, val in labelHeader.items():
                                print(f"    {key} : {val}")
                            print("")
                            asdfasdf
                            newLabelData[..., k] = 1
                        asdfasfd
            print("")
            if isDebug:
                print(f"old={np.sum(oldTime)}, new={np.sum(newTime)}")
                plt.figure()
                plt.imshow(transformed_nrrdData[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255)
                plt.colorbar()
                plt.figure()
                plt.imshow(transformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T, cmap="gray", vmin=0, vmax=255)
                plt.colorbar()
                plt.figure()
                plt.imshow(np.abs(transformed_nrrdData_ref[..., int(transformed_nrrdData.shape[2]/2)].T- transformed_nrrdData[..., int(transformed_nrrdData.shape[2]/2)].T), cmap="gray", vmin=0, vmax=255)
                plt.colorbar()
                plt.show()
            # update header, then save data
            nrrdHeader["sizes"] = np.array(transformed_nrrdData.shape)
            nrrdHeader["space directions"] = np.array([[abs(X[0]-X[1]), 0., 0.], [0., abs(Y[0]-Y[1]), 0.], [0., 0., sliceSpacing]])
            nrrdHeader["space origin"] = np.array([nrrdHeader["space directions"][0, 0] * nrrdHeader["sizes"][0] * 0.5 * (-1), 0.0, nrrdHeader["space directions"][2, 2] * nrrdHeader["sizes"][2] * 0.5 * (-1)])
            nrrdHeader["meta info"] = metadata
            nrrd.write(os.path.join("./transformed_data_mask", os.path.splitext(os.path.basename(nrrdFilePathName))[0] + "_transformed.nrrd"), transformed_nrrdData, nrrdHeader)
            if doOldSC:
                nrrd.write(os.path.join("./transformed_data_mask", os.path.splitext(os.path.basename(nrrdFilePathName))[0] + "_transformed_ref.nrrd"), transformed_nrrdData_ref, nrrdHeader)
            if doLabelUpdateAlso:
                1
        else:
            raise Exception(f"Data and Mask are not matched!!({os.path.basename(nrrdFilePathName).lower().replace('.nrrd', '.seg.nrrd')}!={os.path.basename(labelFilePathName).lower()})")
