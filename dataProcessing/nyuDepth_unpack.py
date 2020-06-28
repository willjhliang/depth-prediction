
import numpy as np
import h5py


with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
    imgs = f.get('images')[()]
    depths = f.get('depths')[()]

m = imgs.shape[0]
imgs = np.moveaxis(imgs, 1, -1)

random = np.arange(m)
np.random.shuffle(random)
imgs = imgs[random]
depths = depths[random]

for i in range(m):
    np.save('nyuDepth/img' + str(i) + '.npy', imgs[i])
    np.save('nyuDepth/dep' + str(i) + '.npy', depths[i])
