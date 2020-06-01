
import math
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

with h5py.File('nyuDepth_v2.mat', 'r') as f:
    imgs = f.get('images')[()]
    depths = f.get('depths')[()]

m = imgs.shape[0]
imgs = np.moveaxis(imgs, 1, -1)
depths = np.expand_dims(depths, axis=-1)

random = np.arange(m)
np.random.shuffle(random)
imgs = imgs[random]
depths = depths[random]


def resize(data, scale):
    ret = []
    for i in range(m):
        img = data[i]
        img = cv2.resize(data[i], (0, 0), fx=.5, fy=.5)
        ret.append(img)
    ret = np.array(ret)
    return ret


imgs = resize(imgs, .4)
depths = resize(depths, .4)


def getSubpics(imgData, depthData, subWidth, subHeight, xStride, yStride):
    imgRet = []
    depthRet = []
    height = imgData.shape[2]
    width = imgData.shape[1]
    print(imgData.shape)
    for i in range(m):
        for j in range(0, height, yStride):
            for k in range(0, width, xStride):
                img = imgData[i, k:k + subWidth, j:j + subHeight, :]
                depth = depthData[i, k:k + subWidth, j:j + subHeight]
                if img.shape[0] != subWidth or img.shape[1] != subHeight:
                    break
                imgRet.append(img)
                depthRet.append(depth)
    imgRet = np.array(imgRet)
    depthRet = np.array(depthRet)
    return imgRet, depthRet


imgs, depths = getSubpics(imgs, depths, 128, 128, 128, 64)

m = imgs.shape[0]
print(imgs.shape)
print(depths.shape)

BATCH_SIZE = 64

for i in range(math.ceil(1.0 * m / BATCH_SIZE)):
    s = BATCH_SIZE * i
    e = min(BATCH_SIZE * (i + 1), m)
    np.savez_compressed('nyuDepth2/data' + str(i) + '.npz',
                        images=imgs[s:e],
                        depths=depths[s:e])
