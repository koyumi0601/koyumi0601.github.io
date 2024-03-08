#! python

import numpy as np
import math
import scipy.interpolate as interp
# import time
# import matplotlib.pyplot as plt

OUTGRIDX = 800
OUTGRIDY = 600

def scanConversion(inputData, info, downscale = 1):
    fovShape = info["FovShape"]
    if(fovShape == "CurvedVector"): 
        scOut = curvedScanConversion(inputData, info, downscale)
    elif(fovShape == "Linear"):
        scOut = linearScanConversion(inputData, info)
    elif(fovShape == "Vector"):
        scOut = phasedScanConversion(inputData, info, downscale)
    scOut = np.array(scOut,dtype="uint8")
    return scOut 

def curvedScanConversion(inputData, info, downscale):
    numGridY = int(OUTGRIDY / downscale)
    numGridX = int(OUTGRIDX / downscale)
    [numSamples, numLines] = inputData.shape
    radiusMm = float(info['radiusOfCurvatureAzimuthMm'])
    displayedLateralMin = float(info['DisplayedLateralMin'])
    displayedLateralSpan =  float(info['DisplayedLateralSpan'])
    displayedAxialMin = float(info['DisplayedAxialMin'])
    displayedAxialSpan = float(info['DisplayedAxialSpan'])
    rangeMmA = np.linspace(radiusMm +  displayedAxialMin, radiusMm + displayedAxialMin + displayedAxialSpan, numSamples)
    thetaA = np.linspace(displayedLateralMin, displayedLateralMin + displayedLateralSpan, numLines)
    initDepthMm = (radiusMm+displayedAxialMin) * math.cos(displayedLateralMin / 180 * np.pi)
    endDepthMm = radiusMm + displayedAxialMin + displayedAxialSpan
    gridY = np.linspace(initDepthMm, endDepthMm, numGridY)
    initAzMm = (radiusMm + displayedAxialMin + displayedAxialSpan) * math.sin(displayedLateralMin / 180 * np.pi)
    endAzMm = (radiusMm + displayedAxialMin + displayedAxialSpan) * math.sin((displayedLateralMin + displayedLateralSpan) / 180 * np.pi)
    gridX = np.linspace(initAzMm, endAzMm, numGridX)
    [mcX, mcY] = np.meshgrid(gridX, gridY)
    scRangeMmA = np.sqrt(mcX**2 + mcY**2)
    scThetaA = np.arctan(mcX / mcY) / np.pi*180
    f = interp.interp2d(thetaA, rangeMmA, inputData, fill_value=0)
    scOut = np.zeros([numGridY, numGridX], dtype="float")
    # startTime = time.time()
    for yIdx in range(numGridY):
        for xIdx in range(numGridX):
            scTheta = scThetaA[yIdx, xIdx]
            scRangeMm = scRangeMmA[yIdx, xIdx]
            if (scTheta >= displayedLateralMin and 
                scTheta <= displayedLateralMin + displayedLateralSpan and 
                scRangeMm >= radiusMm and 
                scRangeMm <= radiusMm + displayedAxialMin + displayedAxialSpan):
                scOut[yIdx, xIdx] = f(scTheta, scRangeMm)
    if not(downscale == 1):
        orgAzimuthA = np.linspace(0, numGridX-1, numGridX)
        orgRangeA = np.linspace(0, numGridY-1, numGridY)
        ipAzimuthA = np.linspace(0, numGridX-1, OUTGRIDX)
        ipRangeA = np.linspace(0, numGridY-1, OUTGRIDY)
        f = interp.interp2d(orgAzimuthA, orgRangeA, scOut, fill_value=0)
        scOut = f(ipAzimuthA, ipRangeA)
    # endTime = time.time()
    # print('ellipse time = %f (sec)' % (endTime - startTime))
    return scOut 

def linearScanConversion(inputData, info):
    numGridY = int(OUTGRIDY)
    numGridX = int(OUTGRIDX)
    [numSamples, numLines] = inputData.shape
    displayedLateralMin =  float(info['DisplayedLateralMin'])
    displayedLateralSpan =  float(info['DisplayedLateralSpan'])
    displayedAxialMin = float(info['DisplayedAxialMin'])
    displayedAxialSpan = float(info['DisplayedAxialSpan'])
    rangeMmA = np.linspace(displayedAxialMin, displayedAxialMin + displayedAxialSpan, numSamples)
    azimuthA = np.linspace(displayedLateralMin, displayedLateralMin + displayedLateralSpan, numLines)
    f = interp.interp2d(azimuthA, rangeMmA, inputData, fill_value=0)
    ipRangeA = np.linspace(displayedAxialMin, displayedAxialMin + displayedAxialSpan, numGridY)
    ipAzimuthA = np.linspace(displayedLateralMin, displayedLateralMin + displayedLateralSpan, numGridX)
    scOut = f(ipAzimuthA, ipRangeA)
    return scOut 

def phasedScanConversion(inputData, info, downscale):
    numGridY = int(OUTGRIDY / downscale)
    numGridX = int(OUTGRIDX / downscale)
    [numSamples, numLines] = inputData.shape
    displayedLateralMin =  float(info['DisplayedLateralMin'])
    displayedLateralSpan =  float(info['DisplayedLateralSpan'])
    displayedAxialMin = float(info['DisplayedAxialMin'])
    displayedAxialSpan = float(info['DisplayedAxialSpan'])
    rangeMmA = np.linspace(displayedAxialMin, displayedAxialMin + displayedAxialSpan, numSamples)
    thetaA = np.linspace(displayedLateralMin, displayedLateralMin + displayedLateralSpan, numLines)
    initDepthMm = displayedAxialMin
    endDepthMm = displayedAxialMin + displayedAxialSpan
    gridY = np.linspace(initDepthMm,endDepthMm,numGridY)
    initAzMm = (displayedAxialMin + displayedAxialSpan) * math.sin(displayedLateralMin / 180 * np.pi)
    endAzMm = (displayedAxialMin + displayedAxialSpan) * math.sin((displayedLateralMin + displayedLateralSpan) / 180 * np.pi)
    gridX = np.linspace(initAzMm, endAzMm, numGridX)
    [mcX, mcY] = np.meshgrid(gridX, gridY)
    scRangeMm = np.sqrt(mcX**2 + mcY**2)
    scThetaA =  np.arctan(mcX / mcY) / np.pi*180
    f = interp.interp2d(thetaA, rangeMmA, inputData, fill_value=0)
    scOut = np.zeros([numGridY, numGridX], dtype="float")
    for yIdx in range(numGridY):
        for xIdx in range(numGridX):
            scOut[yIdx,xIdx] = f(scThetaA[yIdx, xIdx], scRangeMm[yIdx, xIdx])
    if not(downscale == 1):
        orgAzimuthA = np.linspace(0, numGridX-1, numGridX)
        orgRangeA = np.linspace(0, numGridY-1, numGridY)
        ipAzimuthA = np.linspace(0, numGridX-1, OUTGRIDX)
        ipRangeA = np.linspace(0, numGridY-1, OUTGRIDY)
        f = interp.interp2d(orgAzimuthA, orgRangeA, scOut, fill_value=0)
        scOut = f(ipAzimuthA, ipRangeA)
    return scOut 

def importInfo(fileName):
    outputInfo = {}
    with open(fileName) as f:
        for aLine in f.readlines():
            if( aLine.count(':') > 0):
                outputInfo[aLine.split(':')[0].strip()] = aLine.split(':')[1].replace(',', '').strip()
    return outputInfo

def dynamicRange(inputData,pivotIn,pivotOut,dynamicRangeDb,LogCompressStrengthDb):
    dbPerLsb = 255/LogCompressStrengthDb
    dR = dynamicRangeDb * dbPerLsb
    drOut = (255/dR)*(inputData-255*pivotIn) + 255*pivotOut
    drOut = np.clip(drOut,0,255)
    drOut = np.array(drOut,dtype="uint8")
    return drOut

def grayMap(inputData, grayMapIdx, grayMapValue):
    grayFunction = interp.interp1d(grayMapIdx,grayMapValue)
    outPut = grayFunction(inputData)
    return outPut       

def importData(signalFileName,targetFrame,info):
    numLinesPerFrame = int(info['NumLinesPerSlice'])
    numSamplesPerLine = int(info['NumSamplesPerLine'])
    # acquiredAxialMin = float(info['AcquiredAxialMin']) # not used, but could be used for more precise displaying
    acquiredAxialSpan = float(info['AcquiredAxialSpan'])
    # displayedAxialMin = float(info['DisplayedAxialMin']) # not used, but could be used for more precise displaying
    displayedAxialSpan = float(info['DisplayedAxialSpan'])
    samplePerMm = acquiredAxialSpan/numSamplesPerLine
    numDisplaySample = math.floor(displayedAxialSpan/samplePerMm)
    cineData = np.fromfile(signalFileName, dtype=np.uint8)
    countPerFrame= numLinesPerFrame * numSamplesPerLine
    startIdx = (targetFrame-1)*countPerFrame
    endIdx = startIdx + countPerFrame
    acqSignalData = cineData[startIdx:endIdx]
    acqSignalData.shape = [numLinesPerFrame, numSamplesPerLine]
    acqSignalData = np.transpose(acqSignalData)
    displaySignal = acqSignalData[0:numDisplaySample,:]
    return acqSignalData, displaySignal
