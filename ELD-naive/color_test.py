import numpy as np
from matplotlib import pyplot as plt
import h5py
gt_raw = f["data"][1].transpose(1,2,0)
r = gt_raw[:,:,0]#gt_rgb[:,:,0]
g = gt_raw[:,:,1]/2+ gt_raw[:,:,2]/2#gt_rgb[:,:,1]
b = gt_raw[:,:,2]#gt_rgb[:,:,2]
rgb = np.stack([r,g,b], axis=2)
plt.imshow(rgb)
plt.show()