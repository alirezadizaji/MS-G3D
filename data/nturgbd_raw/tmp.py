import os
import numpy as np


for r, _, fs in os.walk('./nturgb+d_skeletons/'):
    for f in fs:
        pth = os.path.join(r, f)
        arr = np.load(pth, allow_pickle=True)
        input(arr.shape)