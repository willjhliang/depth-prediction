
import matplotlib.pyplot as plt
import numpy as np
import cv2


def display(dList):
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(dList)):
        plt.subplot(1, len(dList), i + 1)
        dList[i] = np.swapaxes(dList[i], 0, 1)
        plt.imshow(dList[i])
        plt.axis('off')
    plt.show()
    return fig


img = np.load('../nyuDepth/imgs/img0.npy')
border = 8
img = img[border:img.shape[0] - border, border:img.shape[1] - border]
resized = cv2.resize(img, (0, 0), fx=.6, fy=.6)
crop = img[0:128, 0:128, ...]
flipped = np.flip(crop, 0)

cv2.imwrite('img.png', img)
cv2.imwrite('resized.png', resized)
cv2.imwrite('crop.png', crop)
cv2.imwrite('flipped.png', flipped)
