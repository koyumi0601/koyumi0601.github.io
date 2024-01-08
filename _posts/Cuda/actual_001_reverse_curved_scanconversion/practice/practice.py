import numpy as np

x = np.arange(10)
y = np.arange(20)
z = np.arange(3)
# print(x)
# print(y)
# print(z)

xg, yg = np.meshgrid(x, y)
# print(xg)
# print(yg)

# xg2, yg2, zg2 = np.meshgrid(xg, yg, z)
# print(xg2.shape)

xg3d = np.repeat(xg[:, :, np.newaxis], len(z), axis=2)
yg3d = np.repeat(yg[:, :, np.newaxis], len(z), axis=2)

zg2d = np.repeat(z[:,np.newaxis], len(x), axis=1)
zg3d = np.repeat(zg2d[:,:,np.newaxis], len(y), axis=2)
zg3d_t = zg3d.transpose(1, 2, 0)
print(xg3d[:,:,0])
print(yg3d[:,:,1])
print(zg3d_t[:,:,0])
print(zg3d_t.shape)


# print(zg2d)
# print(np.repeat(z[np.newaxis,:], len(x), axis=0))
# print(np.repeat(z[:, np.newaxis], len(x), axis=1))

# print(np.repeat(zg2d[:, :, np.newaxis], len(x), axis=2))

# print(zg3d)
# print(zg3d.shape)
reshaped_data = np.transpose(zg3d, (1, 2, 0))
# print(zg3d)
# print(zg3d.shape)
# print(reshaped_data[:,:,2])
# print(reshaped_data.shape)
