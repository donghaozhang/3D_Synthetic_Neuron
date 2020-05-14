import os
import numpy as np
from package.rivuletpymaster.rivuletpy.trace import R2Tracer


def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out


tracer = R2Tracer(quality=False, silent=False, speed=False, clean=False, non_stop=False)
prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_full'
allfiles = os.listdir(prefix)
print(allfiles)
for i, file in enumerate(allfiles):
    filepath = os.path.join(prefix, file)
    pred_im = loadtiff3d(filepath) / 255
    swc, soma = tracer.trace(img=pred_im, threshold=0.15)