#! usr/bin/env python

from functions import *
import nrrd, pydicom, natsort, math
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.interpolate import RegularGridInterpolator

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

isSkipStencil = True
if __name__ == "__main__":
    # tstLabelFileRootPathName = select_dir_or_file(statement = "Select a tested nrrd label root directory.", target="dir", isMuliple=False)
    tstLabelFileRootPathName = r"D:\data_of_nrrds\ABUS_KU\Annotation\reconstructed_label-ssl_nrrd"
    tstLabelNrrdFileNameList = globFileList(f"{tstLabelFileRootPathName}/*.nrrd", recursive=True, directoryOnly=False, isFullPath=True)
    # refLabelFileRootPathName = select_dir_or_file(statement = "Select a reference nrrd label root directory.", target="dir", isMuliple=False)
    refLabelFileRootPathName = r"D:\data_of_nrrds\ABUS_KU\Annotation\matched_to_nrrd"
    refLabelFileRootDirName = os.path.basename(refLabelFileRootPathName)
    refLabelNrrdFileNameList = globFileList(f"{refLabelFileRootPathName}/*.nrrd", recursive=True, directoryOnly=False, isFullPath=True)
    print(tstLabelFileRootPathName)
    print(refLabelFileRootPathName)
    aDict = {}
    epsilon = 1e-6
    for tstLabelNrrdFileName in tstLabelNrrdFileNameList:
        baseName = ("_".join(os.path.basename(tstLabelNrrdFileName).split("_")[0:2]))
        tstLabelData, _ = nrrd.read(tstLabelNrrdFileName)
        for refLabelNrrdFileName in refLabelNrrdFileNameList:
            if baseName in refLabelNrrdFileName:
                break
        refLabelData, _ = nrrd.read(refLabelNrrdFileName)
        tstLabelData = tstLabelData.flatten()
        refLabelData = refLabelData.flatten()
        # TP = np.sum((tstLabelData == 0) & (refLabelData == 0))
        # FP = np.sum((tstLabelData == 0) & (refLabelData != 0))
        # TN = np.sum((tstLabelData != 0) & (refLabelData != 0))
        # FN = np.sum((tstLabelData != 0) & (refLabelData == 0))
        # diceScore0 = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
        # TP = np.sum((tstLabelData == 1) & (refLabelData == 1))
        # FP = np.sum((tstLabelData == 1) & (refLabelData != 1))
        # TN = np.sum((tstLabelData != 1) & (refLabelData != 1))
        # FN = np.sum((tstLabelData != 1) & (refLabelData == 1))
        # diceScore1 = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
        # aDict[baseName] = (diceScore0 + diceScore1)/2
        TP = np.sum(np.logical_and(tstLabelData, refLabelData))
        FP = np.sum(np.logical_and(tstLabelData, np.logical_not(refLabelData)))
        FN = np.sum(np.logical_and(np.logical_not(tstLabelData), refLabelData))
        diceScore = (2 * TP) / (2 * TP + FP + FN)

        aDict[baseName] = diceScore
        print(f"{baseName}={aDict[baseName]}")
    print(aDict)
