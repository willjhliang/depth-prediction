
import numpy as np
import h5py
import cv2


with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
    imgs = f.get('images')[()]
    depths = f.get('depths')[()]

m = imgs.shape[0]
imgs = np.moveaxis(imgs, 1, -1)
depths = np.expand_dims(depths, axis=-1)


def resize(data, scale):
    ret = []
    for i in range(m):
        img = data[i]
        img = cv2.resize(data[i], (0, 0), fx=scale, fy=scale)
        ret.append(img)
    ret = np.array(ret)
    return ret


imgs = resize(imgs, .5)
depths = resize(depths, .5)

for i in range(m):
    np.save('nyuDepth2/img' + str(i) + '.npy', imgs[i])
    np.save('nyuDepth2/dep' + str(i) + '.npy', depths[i])
