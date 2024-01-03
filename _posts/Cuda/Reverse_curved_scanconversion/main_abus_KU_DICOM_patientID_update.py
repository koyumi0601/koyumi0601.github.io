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
        desc = str(dicomObj[key].description())
        if desc.lower() == targetDesc.lower():
            return str(dicomObj[key].value)

def get_Tag_Val_By_addr(dicomObj, targetAddr):
    for tag, key in zip(dicomObj, dicomObj.keys()):
        if tag.tag == targetAddr:
            return dicomObj[key].value


def get_item_value(ds, tagList):
    v = ds
    try:
        for idx, tag in enumerate(tagList):
            if idx == len(tagList) - 1:
                return v[tag].value
            else:
                v = v[tag].value[0]
    except:
        return None

def set_tag_value(ds, tag, value):
    """
    Set the value of a DICOM tag using its address.

    Parameters:
    - ds: pydicom Dataset object
    - tag: Tuple representing the DICOM tag (group, element)
    - value: New value for the tag
    """
    tag_address = pydicom.tag.Tag(*tag)
    # Check if the tag exists in the dataset
    if tag_address in ds:
        data_element = ds[tag_address]
        data_element.value = value
    else:
        # If the tag doesn't exist, create it
        crashed
        ds.add_new(tag_address, '??', value)



Patient_ID = (0x0010, 0x0020)
Series_Description = (0x0008, 0x103E)

if __name__ == "__main__":
    dcmFilePathNameSelected = select_dir_or_file(statement = "Select a target pair DICOM directory.", target="dir", isMuliple=False)
    dicomFileNameList = globFileList(f"{dcmFilePathNameSelected}/*.*", recursive=True, directoryOnly=False, isFullPath=True)
    
    usDicomFileNameList = natsort.natsorted([a for a in dicomFileNameList if "\\US\\" in a ])
    usDataNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[2].replace("-", "") for a in usDicomFileNameList]
    usPriorDirNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[1] for a in usDicomFileNameList]

    srDicomFileNameList = natsort.natsorted([a for a in dicomFileNameList if "\\SR\\" in a ])
    srDataNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[2].replace("-", "") for a in srDicomFileNameList]
    srPriorDirNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[1] for a in srDicomFileNameList]

    rpDicomFileNameList = globFileList(f"{dcmFilePathNameSelected}/*.dcm", recursive=True, directoryOnly=False, isFullPath=True)
    rpDataNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[2].replace("-", "") for a in rpDicomFileNameList]
    rpPriorDirNameList = [a.replace("\\", "/").split(dcmFilePathNameSelected)[-1].split("/")[1] for a in rpDicomFileNameList]

    for usDicomFileName, usPriorDirName, usDataName in zip(usDicomFileNameList, usPriorDirNameList, usDataNameList):
        ds_US = pydicom.read_file(usDicomFileName, stop_before_pixels=False)
        # get DICOM tag
        dicomViewName = get_item_value(ds_US, [Series_Description])
        usDataName = f"{usDataName.upper()}_{dicomViewName.upper()}"
        # update DICOM tag
        patientID = get_item_value(ds_US, [Patient_ID])
        set_tag_value(ds_US, Patient_ID, f"{usDataName}_{patientID}")
        # save DICOM
        ds_US.save_as(usDicomFileName)
        print(f"Updating US data {usDataName}...")
    for srDicomFileName, srPriorDirName, srDataName in zip(srDicomFileNameList, srPriorDirNameList, srDataNameList):
        ds_sr = pydicom.read_file(srDicomFileName, stop_before_pixels=False)
        # get DICOM tag
        srDataName = f"{srDataName.upper()}"
        # update DICOM tag
        patientID = get_item_value(ds_sr, [Patient_ID])
        set_tag_value(ds_sr, Patient_ID, f"{srDataName}_{patientID}")
        # save DICOM
        ds_sr.save_as(srDicomFileName)
        print(f"Updating SR data {srDataName}...")
    for rpDicomFileName, rpPriorDirName, rpDataName in zip(rpDicomFileNameList, rpPriorDirNameList, rpDataNameList):
        ds_rp = pydicom.read_file(rpDicomFileName, stop_before_pixels=False)
        # get DICOM tag
        rpDataName = f"{rpDataName.upper()}"
        # update DICOM tag
        patientID = get_item_value(ds_rp, [Patient_ID])
        set_tag_value(ds_rp, Patient_ID, f"{rpDataName}_{patientID}")
        # save DICOM
        ds_rp.save_as(rpDicomFileName)
        print(f"Updating RP data {rpDataName}...")
