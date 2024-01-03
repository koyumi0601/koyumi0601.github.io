#! usr/bin/env python

from functions import *
import nrrd, pydicom, natsort
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos
import time
from collections import OrderedDict
import io
import PIL.Image as Image
import zlib


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

def get_uncompressed_dicom_tag(ds, targetAddr):
    compressed_data = ds[targetAddr].value
    uncompressed_data = zlib.decompress(compressed_data)
    return uncompressed_data

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


def save_bytes_to_file(byte_data, file_path):
    with open(file_path, "wb") as file:
        file.write(byte_data)


def uint8_to_binary_list(uint8_values):
    binary_list = []
    for value in uint8_values:
        binary_string = bin(value)[2:].zfill(8)
        binary_list.extend([int(bit) for bit in binary_string])
    return binary_list


Patient_ID = (0x0010, 0x0020)
Series_Description = (0x0008, 0x103E)
Breast_Border_Geometry = (0x0021, 0x1100)
Chest_Wall_Geometry = (0x0021, 0x1110)
Rib_Geometry = (0x0021, 0x1120)

Row = (0x0028, 0x0010)
Column = (0x0028, 0x0011)
Depth = (0x0028, 0x0008)

import numpy as np


# binary_data = b'Hello, \ASCII!'
# try:
#     ascii_text = binary_data.decode('ascii')
#     print(ascii_text)
# except UnicodeDecodeError as e:
#     print(f"Error: {e}")
# asdf

# binary_data = b'\x00\x00\x80\x40'  # 이진 표현: 1.0
# float_value = np.frombuffer(binary_data, dtype=np.float32)[0]
# sign_bit = (float_value.view(np.int32) >> 31) & 0x01
# exponent = (float_value.view(np.int32) >> 23) & 0xFF
# mantissa = float_value.view(np.int32) & 0x7FFFFF
# exponent_bits = 15
# mantissa_bits = 17
# sign_bit_str = format(sign_bit, '0b')
# exponent_str = format(exponent, f'0{exponent_bits}b')
# mantissa_str = format(mantissa, f'0{mantissa_bits}b')
# print(f"Sign: {sign_bit_str}, Exponent: {exponent_str}, Mantissa: {mantissa_str}")
# print(float_value)

# head = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\x00'

# numNum = 10
# endian = "<"

# dt = np.dtype("uint8")
# dt = dt.newbyteorder(endian)
# print(f"uint8, {np.frombuffer(head[0:1*numNum], dtype=dt)}")
# dt = np.dtype("int8")
# dt = dt.newbyteorder(endian)
# print(f"int8, {np.frombuffer(head[0:1*numNum], dtype=dt)}")
# dt = np.dtype("int16")
# dt = dt.newbyteorder(endian)
# print(f"int16, {np.frombuffer(head[0:2*numNum], dtype=dt)}")
# dt = np.dtype("uint16")
# dt = dt.newbyteorder(endian)
# print(f"uint16, {np.frombuffer(head[0:2*numNum], dtype=dt)}")
# dt = np.dtype("int32")
# dt = dt.newbyteorder(endian)
# print(f"int32, {np.frombuffer(head[0:4*numNum], dtype=dt)}")
# dt = np.dtype("uint32")
# dt = dt.newbyteorder(endian)
# print(f"uint32, {np.frombuffer(head[0:4*numNum], dtype=dt)}")
# dt = np.dtype("float32")
# dt = dt.newbyteorder(endian)
# print(f"float32, {np.frombuffer(head[0:4*numNum], dtype=dt)}")
# dt = np.dtype("int64")
# dt = dt.newbyteorder(endian)
# print(f"int64, {np.frombuffer(head[0:8], dtype='int64')}")
# dt = np.dtype("uint64")
# dt = dt.newbyteorder(endian)
# print(f"uint64, {np.frombuffer(head[0:8], dtype='uint64')}")





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

    aList0 = []
    aList1 = []
    aList2 = []
    aList3 = []
    head = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\x00'

    headerLength = 10

    for idx, (usDicomFileName, usPriorDirName, usDataName) in enumerate(zip(usDicomFileNameList, usPriorDirNameList, usDataNameList)):
        ds_US = pydicom.read_file(usDicomFileName, stop_before_pixels=False)

        # get DICOM tag1
        dicomViewName = get_item_value(ds_US, [Series_Description])
        usDataName = f"{usDataName.upper()}_{dicomViewName.upper()}"
        # get DICOM tag2
        breastBorderGeometry = get_item_value(ds_US, [Breast_Border_Geometry])
        chestWallGeometry = get_item_value(ds_US, [Chest_Wall_Geometry])
        row = get_item_value(ds_US, [Row])
        column = get_item_value(ds_US, [Column])
        depth = get_item_value(ds_US, [Depth])
        



        print(f"Checking US data {usDataName}...")
        # print(f"    {breastBorderGeometry[0:54]}")
        hex_representation = ''.join([f'\\x{byte:02x}' for byte in breastBorderGeometry])
        print(f"    {hex_representation[0:216]}")
        print(f"    Bytes of breastBorderGeometry = {len(breastBorderGeometry)}")
        print(f"    US_volume_dimension = {row} x {column} x {depth}")

        # ndarray_data_int16 = np.frombuffer(breastBorderGeometry, dtype="int16")
        # ndarray_data_int32 = np.frombuffer(breastBorderGeometry, dtype="int32")
        # ndarray_data_float = np.frombuffer(breastBorderGeometry, dtype=float)



        endian = "<"
        dt = np.dtype("uint8")
        dt = dt.newbyteorder(endian)
        ndarray_data_uint8 = np.frombuffer(breastBorderGeometry[headerLength:320000+headerLength], dtype=dt)
        img1 = ndarray_data_uint8[0:160000].reshape(400,400)
        img2 = ndarray_data_uint8[160000:].reshape(400,400)
        # plt.figure()
        # plt.imshow(img1, cmap="gray", vmin=0, vmax=255)
        # plt.figure()
        # plt.imshow(img2, cmap="gray", vmin=0, vmax=255)
        # plt.show()
        # print(f"img1 min = {np.min(img1)}, max = {np.max(img1)}")
        # print(f"img2 min = {np.min(img2)}, max = {np.max(img2)}")

        dt = np.dtype("uint16")
        dt = dt.newbyteorder(endian)
        ndarray_data_uint16 = np.frombuffer(breastBorderGeometry[headerLength:320000+headerLength], dtype=dt)
        pair1 = ndarray_data_uint16[0:80000]
        pair2 = ndarray_data_uint16[80000:]

        print(ndarray_data_uint8[0:10])
        print(ndarray_data_uint16[0:10])

        asdf


        # import cv2
        # cv2.imwrite("image1.png", ndarray_data_uint8[0:160000].reshape(400,400))
        # cv2.imwrite("image2.png", ndarray_data_uint8[160000:].reshape(400,400))
        
        
        # print(ndarray_data_int16[0:30])
        # print(ndarray_data_int32[0:30])
        # print(ndarray_data_float[0:30])
        

    #     print("")
    #     aList0.append(column*depth/len(breastBorderGeometry))
    #     aList1.append(column*depth)
    #     aList2.append(breastBorderGeometry[0:10] == head)
    #     for byteIdx in range(0,len(breastBorderGeometry)-len(head)):
    #         if breastBorderGeometry[byteIdx:byteIdx+len(head)] == head:
    #             aList3.append(byteIdx)
    #     print(aList3)
    #     asdf
    #     if idx == 21111:
    #         break
    # print(aList2)

    # print(aList)
    # for aValue in aList1:
    #     print(aValue)

        # ndarray_data = np.frombuffer(breastBorderGeometry, dtype="uint8")
        # ndarray_byte_data = (np.array(uint8_to_binary_list(ndarray_data.astype("uint8"))).astype("uint8") * 255).tobytes()
        # save_bytes_to_file(ndarray_byte_data, "test3.bin")
