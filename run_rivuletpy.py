import os
import numpy as np
from package.rivuletpymaster.rivuletpy.trace import R2Tracer
import matplotlib.pyplot as plt


def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out


tracer = R2Tracer(quality=False, silent=True, speed=False, clean=False, non_stop=False)
prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_full'
allfiles = os.listdir(prefix)
print(allfiles)
for i, file in enumerate(allfiles):
    if file.endswith('.tif'):
        filepath = os.path.join(prefix, file)
        pred_im = loadtiff3d(filepath)
        pred_im_2d = np.max(pred_im, axis=0)
        fig = plt.figure()
        plt.imshow(pred_im_2d)
        tracer = R2Tracer(quality=False, silent=True, speed=False, clean=False, non_stop=False)
        swc, soma = tracer.trace(img=pred_im, threshold=8)
        swcarray = swc.get_array()
        #swc N x 7 0 sampleid 1 typeid 2 xloc 3 yloc 4 zloc 5 radius 6 parentid
        xloc = swcarray
        swc.save(fname=filepath+'.swc')
# test


