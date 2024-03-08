import matplotlib.pyplot as plt
import backendProcessing.backendProcessing as BE

#fullPathName = r"D:\02.TFS_programs\UE_programs\UltraLabs\ShowCineImage\exampledatas\Convex_6C1\BCine-0-49-Image-20220818_072712.img"
fullPathName = r"D:\external_hdd_important\Backup_20230731\Work\TFS\UE_Tools_Deploy\UltraLabs\showCineImage\exampledatas\Convex_6C1\BCine-0-49-Image-20220818_072712.img"
infoFileName = fullPathName.replace('.img','.trsc')

targetFrame = 30
LogCompressStrengthDb = 96
dynamicRange = 70
pivotIn = 0.5
pivotOut = 0.4
grayMapIdx = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255]
grayMapValue = [0,3,6,10,13,16,20,23,27,30,34,38,42,46,50,54,59,63,68,73,78,83,89,94,99,105,110,115,120,125,130,135,140,146,151,157,163,168,174,181,187,194,200,207,213,220,226,233,239,245,250,255]

info = BE.importInfo(infoFileName)
acqSignalData, displaySignal = BE.importData(fullPathName, targetFrame, info)
drOut = BE.dynamicRange(displaySignal, pivotIn, pivotOut, dynamicRange, LogCompressStrengthDb)
grayMapOut = BE.grayMap(drOut, grayMapIdx, grayMapValue)
scOut = BE.scanConversion(grayMapOut, info, downscale = 2)

plt.figure(1)
plt.imshow(displaySignal, cmap="gray")
plt.figure(2)
plt.imshow(scOut, cmap="gray")
plt.show()
