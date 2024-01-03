#! usr/bin/env python

from functions import *
import nrrd, pydicom, natsort, math
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy

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
        desc = str(dicomObj[key].description())
        if desc.lower() == targetDesc.lower():
            return str(dicomObj[key].value)

def get_Tag_Val_By_addr(dicomObj, targetAddr):
    for tag, key in zip(dicomObj, dicomObj.keys()):
        if tag.tag == targetAddr:
            return dicomObj[key].value

def find_first_last_indices(arr):
    indices = np.where(arr == 1)[0]
    if len(indices) == 0:
        return None, None
    else:
        first_index = indices[0]
        last_index = indices[-1]
        return str(first_index - 1), str(last_index + 1)

isSkipStencil = True
if __name__ == "__main__":

    labelFilePathNameList = select_dir_or_file(statement = "Select a target nrrd label files.", target="file", isMuliple=True)
    nrrdFilePathNameSelected = select_dir_or_file(statement = "Select a target nrrd directory.", target="dir", isMuliple=False)
    nrrdFileNameList = globFileList(f"{nrrdFilePathNameSelected}/*.nrrd", recursive=True, directoryOnly=False, isFullPath=True)

    # labelFilePathNameList = ["D:/works/Reverse_curved_scanconversion/matched_to_DICOM/test/ABUS053_RLAT.seg.nrrd"]
    # nrrdFilePathNameSelected = r"D:/works/Reverse_curved_scanconversion/transformed_data_mask_KU"
    # nrrdFileNameList = globFileList(f"{nrrdFilePathNameSelected}/*.nrrd", recursive=True, directoryOnly=False, isFullPath=True)

    for num, labelFilePathName in enumerate(labelFilePathNameList):

        labelData, labelHeader= nrrd.read(labelFilePathName)

        priorDirName = labelFilePathName.split("/")[-2]
        labelFileName = labelFilePathName.split("/")[-1]
        labelDataName = labelFileName.split(".")[0]

        nrrdFilePathName = ''
        for nrrdFileName in nrrdFileNameList:
            if labelDataName in nrrdFileName:
                nrrdFilePathName = nrrdFileName

        _, nrrdHeader= nrrd.read(nrrdFilePathName)
        # print("")
        # for key, val in labelHeader.items():
        #     print(f"{key}: {val}")
        # print("")

        classList = np.unique(labelData).tolist()
        classList.remove(0)


        if (labelHeader["sizes"][2] == nrrdHeader["sizes"][2]):
            X = [labelHeader["space origin"][0]]
            for idx in range(labelHeader["sizes"][0]-1):
                X.append(X[-1] + labelHeader["space directions"][0, 0])
            Y = [labelHeader["space origin"][1]]
            for idx in range(labelHeader["sizes"][1]-1):
                Y.append(Y[-1] + labelHeader["space directions"][1, 1])
            XX = [nrrdHeader["space origin"][0]]
            for idx in range(nrrdHeader["sizes"][0]-1):
                XX.append(XX[-1] + nrrdHeader["space directions"][0, 0])
            YY = [nrrdHeader["space origin"][1]]
            for idx in range(nrrdHeader["sizes"][1]-1):
                YY.append(YY[-1] + nrrdHeader["space directions"][1, 1])
            XXi, YYi = np.meshgrid(XX, YY, indexing='ij')

            accumedNewLabelData = np.zeros(nrrdHeader["sizes"], dtype=dtypeDict[labelHeader["type"]])
            for segNum, classVal in enumerate(classList):
                print(f"    Class: {segNum}")
                extent = np.array(labelHeader[f"Segment{segNum}_Extent"].split(" ")).astype('float')
                labelHeader[f"Segment{segNum}_Extent"] = " ".join([str(0), str(accumedNewLabelData.shape[0]-1),
                                                                   str(0), str(accumedNewLabelData.shape[1]-1),
                                                                   str(0), str(accumedNewLabelData.shape[2]-1)])
                # newExtent = " ".join(np.round(extent[0:2] * labelHeader["space directions"][0, 0] / nrrdHeader["space directions"][0, 0]).astype(int).astype(str).tolist() + 
                #                      np.round(extent[2:4] * labelHeader["space directions"][1, 1] / nrrdHeader["space directions"][1, 1]).astype(int).astype(str).tolist() + 
                #                      extent[4:6].astype(int).astype(str).tolist())
                # labelHeader[f"Segment{segNum}_Extent"] = newExtent

                newLabelData = np.zeros(nrrdHeader["sizes"], dtype=dtypeDict[labelHeader["type"]])
                for s in range(labelHeader["sizes"][2]):
                    tmpLabel = deepcopy(labelData[..., s])
                    tmpLabel[tmpLabel != classVal] = 0
                    tmpLabel[tmpLabel == classVal] = 1
                    interp_func = RegularGridInterpolator((X, Y), tmpLabel, method='linear', bounds_error=False, fill_value=0)
                    transformed_label = interp_func((XXi, YYi))
                    transformed_label[np.isnan(transformed_label)] = 0
                    transformed_label[transformed_label >= 0.5] = classVal
                    transformed_label[transformed_label < 0.5] = 0
                    newLabelData[..., s] = transformed_label
                newLabelData = newLabelData.astype(dtypeDict[labelHeader["type"]])

                # print(labelHeader[f"Segment{segNum}_Extent"])
                # newExtent = " ".join(list(find_first_last_indices(np.sum(np.sum(newLabelData, 2), 1).astype(bool).astype(int))) + 
                #                      list(find_first_last_indices(np.sum(np.sum(newLabelData, 2), 0).astype(bool).astype(int))) + 
                #                      [str(int(extent[4] - 1)), str(int(extent[5]))])
                # labelHeader[f"Segment{segNum}_Extent"] = newExtent
                # print(labelHeader[f"Segment{segNum}_Extent"])

                accumedNewLabelData = accumedNewLabelData + newLabelData
            accumedNewLabelData = accumedNewLabelData.astype(dtypeDict[labelHeader["type"]])
            del labelHeader["Segmentation_ConversionParameters"]
            labelHeader["sizes"] = nrrdHeader["sizes"]
            labelHeader["space origin"] = nrrdHeader["space origin"]
            labelHeader["space directions"] = nrrdHeader["space directions"]

            newLabelData = np.zeros_like(accumedNewLabelData)
            newLabelData[:,:,0:accumedNewLabelData.shape[2]-1] = accumedNewLabelData[:,:,1:accumedNewLabelData.shape[2]]
            os.makedirs("./matched_to_nrrd", exist_ok = True)
            os.makedirs(f"./matched_to_nrrd/{priorDirName}", exist_ok = True)
            nrrd.write(f"./matched_to_nrrd/{priorDirName}/{labelFileName}", newLabelData, labelHeader)
            print(f"    {labelFilePathName} is processed ({num+1}/{len(labelFilePathNameList)})...")
            print("")
