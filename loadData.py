
import numpy as np
import matplotlib.pyplot as plt


def display(displayList):
    plt.figure(figsize=(15, 15))
    for i in range(len(displayList)):
        plt.subplot(1, len(displayList), i + 1)
        plt.imshow(displayList[i])
        plt.axis('off')
    plt.show()


data = np.load('nyuDepth/data0.npz')
imgs = data['images']
deps = data['depths']

print(imgs.shape)
display([imgs[0], deps[0]])
