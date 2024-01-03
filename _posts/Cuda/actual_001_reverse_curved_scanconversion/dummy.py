#! usr/bin/env python

from functions import *
import nrrd
import matplotlib.pyplot as plt

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

# setting of geometry paramters
metadata = {'PixelSpacing' : (0.200, 0.073),
            'SpacingBetweenSlices': 0.475674,
            'CurvatureRadiusProbe' : 442.11,
            'MaxCut' : 0.0450771}

dataFilePathNameList = select_dir_or_file(statement = "Select a target nrrd data file.", target="file", select_filter="nrrd File (*.nrrd)", isMuliple=True)
for dataFilePathName in dataFilePathNameList:
    _, nrrdHeader= nrrd.read(dataFilePathName)
    print(nrrdHeader["space directions"])
    print(nrrdHeader["space origin"])

    # print(f'{nrrdHeader["space directions"][0,0]}, {nrrdHeader["space directions"][1,1]}')