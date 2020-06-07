
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('dark_background')


def display(displayList):
    plt.figure(figsize=(15, 15))
    for i in range(len(displayList)):
        plt.subplot(1, len(displayList), i + 1)
        plt.imshow(displayList[i])
        plt.axis('off')
    plt.show()


img = np.load('nyuDepth/img0.npy')
dep = np.load('nyuDepth/dep0.npy')

print(img.shape)
print(dep.shape)

display([img, dep])
