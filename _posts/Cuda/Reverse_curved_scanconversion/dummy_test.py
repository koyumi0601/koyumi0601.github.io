import nrrd
import numpy as np
import matplotlib.pyplot as plt

print("[Slow]")
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS05_LUOQ.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS17_RAP.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS17_RLAT.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS17_RMED.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS43_RAP.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS53_RAP.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
print("")
plt.figure()
plt.imshow(stillcut, cmap="gray")


print("[Fast]")
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS01_LAP.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS39_LAP.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
nrrdData, nrrdHeader= nrrd.read(r"D:\works\Reverse_curved_scanconversion\Data\ABUS41_LMED.nrrd")
stillcut = nrrdData[:,:, int(nrrdData.shape[2]/2)]
print(nrrdData.shape, np.min(nrrdData[:,:, int(nrrdData.shape[2]/2)]), np.max(nrrdData[:,:, int(nrrdData.shape[2]/2)]), len(stillcut[stillcut!=0]))
print("")
plt.figure()
plt.imshow(stillcut, cmap="gray")
plt.show()