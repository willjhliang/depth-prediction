
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

with h5py.File('nyuDepth_v2.mat', 'r') as f:
    imgs = f.get('images')[()]
    depths = f.get('depths')[()]

m = imgs.shape[0]


def resize(data, scale):
    ret = []
    for i in range(m):
        img = data[i]
        img = cv2.resize(data[i], (0, 0), fx=.5, fy=.5)
        ret.append(img)
    ret = np.array(ret)
    return ret


imgs = np.moveaxis(imgs, 1, -1)
depths = np.expand_dims(depths, axis=-1)

imgs = resize(imgs, .5)
depths = resize(depths, .5)


def display(displayList):
    plt.figure(figsize=(15, 15))
    for i in range(len(displayList)):
        plt.subplot(1, len(displayList), i + 1)
        plt.imshow(displayList[i])
        plt.axis('off')
    plt.show()


def getSubpics(imgData, depthData, lenRatio, strideRatio):
    imgRet = []
    depthRet = []
    height = imgData.shape[2]
    width = imgData.shape[1]
    subHeight = (int)(height * lenRatio)
    subWidth = (int)(width * lenRatio)
    yStride = (int)(height * strideRatio)
    xStride = (int)(width * strideRatio)
    print(yStride)
    for i in range(m):
        for j in range(0, height, yStride):
            for k in range(0, width, xStride):
                img = imgData[i, k:k + subWidth, j:j + subHeight, :]
                depth = depthData[i, k:k + subWidth, j:j + subHeight]
                if img.shape[0] != subWidth or img.shape[1] != subHeight:
                    continue
                imgRet.append(img)
                depthRet.append(depth)
    imgRet = np.array(imgRet)
    depthRet = np.array(depthRet)
    return imgRet, depthRet


imgs, depths = getSubpics(imgs, depths, .75, .25)

print(imgs.shape)
print(depths.shape)

np.savez_compressed('data.npz', images=imgs, depths=depths)
