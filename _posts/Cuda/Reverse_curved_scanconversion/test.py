#! usr/bin/env python

from functions import *
import nrrd
import matplotlib.pyplot as plt
from math import sin, cos
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

# setting of geometry paramters
metadata = {'PixelSpacing' : (0.200, 0.073),
            'SpacingBetweenSlices': 0.475674,
            'CurvatureRadiusProbe' : 442.11,
            'MaxCut' : 0.0450771}

targetPrecision = 0.05
targetResamplingUnitLength = 0.150

if __name__ == "__main__":
    dataFilePathNameList = select_dir_or_file(statement = "Select a target nrrd data file.", target="file", select_filter="nrrd File (*.nrrd)", isMuliple=True)
    sgmtFilePathNameList = select_dir_or_file(statement = "Select a target nrrd segmentaion file.", target="file", select_filter="nrrd File (*.nrrd)", isMuliple=True)
    for num, (dataFilePathName, sgmtFilePathName) in enumerate(zip(dataFilePathNameList, sgmtFilePathNameList)):
        if os.path.basename(dataFilePathName).lower().replace("data", "mask") == os.path.basename(sgmtFilePathName).lower():
            nrrdData, nrrdHeader = nrrd.read(dataFilePathName)
            nrrdSgmt, nrrdSgmtHeader = nrrd.read(sgmtFilePathName)
            tmpData = nrrdData[..., int(nrrdHeader["sizes"][2]/2)]

            # print("")
            # for key, val in nrrdSgmtHeader.items():
            #     print(f"{key}: {val}")
            # print("")
            # ASDFAF

            # scanconversion params
            rangeSpacing = metadata['PixelSpacing'][1]
            azimuthalSpacing = metadata['PixelSpacing'][0]
            sliceSpacing = metadata['SpacingBetweenSlices']
            probeRadius = metadata['CurvatureRadiusProbe']
            maxCut = metadata['MaxCut']

            # make stencil mask
            clip_size = np.round(np.linspace(np.round(tmpData.shape[0] * maxCut), 0, tmpData.shape[1]))
            stencil_mask = np.ones((tmpData.shape[0], tmpData.shape[1]))
            for idx, clip in zip(range(stencil_mask.shape[1]), clip_size):
                if clip != 0:
                    stencil_mask[:int(clip), idx] = 0
                    stencil_mask[-int(clip):, idx] = 0
            stenciled_nrrdData = np.zeros_like(nrrdData)
            for s in range(stenciled_nrrdData.shape[2]):
                stenciled_nrrdData[:, :, s] = nrrdData[:, :, s] * stencil_mask

            # set RA coordinate of src data : R = srcRangeA (mm), A = srcAngleA (degree)
            tmpData = nrrdData[..., int(nrrdHeader["sizes"][2]/2)]
            center_IJK = [(tmpData.shape[0]-1) / 2.0, 0]
            srcRangeA = probeRadius - (np.arange(tmpData.shape[1]) - center_IJK[1]) * rangeSpacing
            srcAngleA = (np.arange(tmpData.shape[0]) - center_IJK[0]) * azimuthalSpacing / probeRadius * 180 / np.pi

            # set Cart coordinate for destinated data
            meanSampling = (azimuthalSpacing + rangeSpacing)/2.0
            # targetResamplingUnitLength = np.round(np.round(meanSampling / targetPrecision) * targetPrecision, 2)
            targetXMin = (-1) * np.round(np.round(np.abs(max(srcRangeA) * sin(min(srcAngleA) / 180.0 * np.pi)) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetXMax = np.round(np.round(max(srcRangeA) * sin(max(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetYMin = np.round(np.round((min(srcRangeA)) * cos(min(srcAngleA) / 180.0 * np.pi) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)
            targetYMax = np.round(np.round(max(srcRangeA) / targetResamplingUnitLength) * targetResamplingUnitLength, 2)

            X = [targetXMin]
            while(True):
                if X[-1] + targetResamplingUnitLength > targetXMax:
                    break
                else:
                    X.append(X[-1] + targetResamplingUnitLength)
            Y = [targetYMin]
            while(True):
                if Y[-1] + targetResamplingUnitLength > targetYMax:
                    break
                else:
                    Y.append(Y[-1] + targetResamplingUnitLength)

            Xi, Yi = np.meshgrid(X, Y, indexing='ij')
            dstRangeMesh = np.sqrt(Xi**2 + Yi**2)
            dstAngleMesh = np.arctan(Xi / Yi) / np.pi * 180.0

            transformed_nrrdData = np.zeros((len(X), len(Y), stenciled_nrrdData.shape[2]), dtype=dtypeDict[nrrdHeader["type"]])
            transformed_nrrdMaskData = np.zeros((len(X), len(Y), stenciled_nrrdData.shape[2]), dtype=dtypeDict[nrrdSgmtHeader["type"]])
            for s in range(stenciled_nrrdData.shape[2]):
                # data
                tmpData = stenciled_nrrdData[..., s]
                interp_func = RegularGridInterpolator((srcAngleA, srcRangeA), tmpData, method='linear', bounds_error=False, fill_value=0)
                transformed = np.fliplr(interp_func((dstAngleMesh, dstRangeMesh)))
                transformed[np.isnan(transformed)] = 0
                transformed = transformed.astype(dtypeDict[nrrdHeader["type"]])
                transformed_nrrdData[..., s] = transformed

                # segmentation
                tmpSgmt = nrrdSgmt[..., s]
                interp_func = RegularGridInterpolator((srcAngleA, srcRangeA), tmpSgmt, method='linear', bounds_error=False, fill_value=0)
                transformed_mask = np.fliplr(interp_func((dstAngleMesh, dstRangeMesh)))
                transformed_mask[np.isnan(transformed_mask)] = 0
                transformed_mask[transformed_mask >= 0.5] = 1
                transformed_mask[transformed_mask < 0.5] = 0
                transformed_mask = transformed_mask.astype(dtypeDict[nrrdSgmtHeader["type"]])
                transformed_nrrdMaskData[..., s] = transformed_mask

            # update header , then save data
            os.makedirs(f"./transformed_data_mask_TDSC", exist_ok=True)
            nrrdHeader["sizes"] = np.array(transformed_nrrdData.shape)
            nrrdHeader["space directions"] = np.array([[targetResamplingUnitLength, 0., 0.], [0., targetResamplingUnitLength, 0.], [0., 0., sliceSpacing]])
            nrrdHeader["space origin"] = np.array([nrrdHeader["space directions"][0, 0] * (nrrdHeader["sizes"][0] - 1) * 0.5 * (-1), 0.0, nrrdHeader["space directions"][2, 2] * (nrrdHeader["sizes"][2] - 1) * 0.5 * (-1)])
            nrrdHeader["meta info"] = metadata
            nrrd.write(os.path.join("./transformed_data_mask_TDSC", os.path.splitext(os.path.basename(dataFilePathName))[0] + "_transformed.nrrd"), transformed_nrrdData, nrrdHeader)
            nrrdSgmtHeader["sizes"] = np.array(transformed_nrrdMaskData.shape)
            nrrdSgmtHeader["space directions"] = np.array([[targetResamplingUnitLength, 0., 0.], [0., targetResamplingUnitLength, 0.], [0., 0., sliceSpacing]])
            nrrdSgmtHeader["space origin"] = np.array([nrrdSgmtHeader["space directions"][0, 0] * (nrrdSgmtHeader["sizes"][0] - 1) * 0.5 * (-1), 0.0, nrrdSgmtHeader["space directions"][2, 2] * (nrrdSgmtHeader["sizes"][2] - 1) * 0.5 * (-1)])
            nrrdSgmtHeader["meta info"] = metadata
            nrrd.write(os.path.join("./transformed_data_mask_TDSC", os.path.splitext(os.path.basename(sgmtFilePathName))[0] + "_transformed.nrrd"), transformed_nrrdMaskData, nrrdSgmtHeader)
            print(f"    {dataFilePathName} is processed ({num+1}/{len(dataFilePathNameList)})...")
        else:
            raise Exception(f"Data and Mask are not matched!!({os.path.basename(dataFilePathName).lower().replace('data', 'mask')}!={os.path.basename(sgmtFilePathName).lower()})")

