import os
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from package.rivuletpymaster.rivuletpy.trace import R2Tracer


def loadswc(filepath):
    '''
    Load swc file as a N X 7 numpy array
    '''
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                if len(cells) ==7:
                    cells = [float(c) for c in cells]
                    swc.append(cells)
    return np.array(swc)


def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out


def precision_recall(swc1, swc2, dist1=4, dist2=4):
    '''
    Calculate the precision, recall and F1 score between swc1 and swc2 (ground truth)
    It generates a new swc file with node types indicating the agreement between two input swc files
    In the output swc file: node type - 1. the node is in both swc1 agree with swc2
                                                        - 2. the node is in swc1, not in swc2 (over-traced)
                                                        - 3. the node is in swc2, not in swc1 (under-traced)
    target: The swc from the tracing method
    gt: The swc from the ground truth
    dist1: The distance to consider for precision
    dist2: The distance to consider for recall
    '''

    TPCOLOUR, FPCOLOUR, FNCOLOUR  = 3, 2, 180 # COLOUR is the SWC node type defined for visualising in V3D

    d = cdist(swc1[:, 2:5], swc2[:, 2:5])
    mindist1 = d.min(axis=1)
    tp = (mindist1 < dist1).sum()
    fp = swc1.shape[0] - tp

    mindist2 = d.min(axis=0)
    fn = (mindist2 > dist2).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Make the swc for visual comparison
    swc1[mindist1 <= dist1, 1] = TPCOLOUR
    swc1[mindist1 > dist1, 1] = FPCOLOUR
    swc2_fn = swc2[mindist2 > dist2, :]
    swc2_fn[:, 0] = swc2_fn[:, 0] + 100000
    swc2_fn[:, -1] = swc2_fn[:, -1] + 100000
    swc2_fn[:, 1] = FNCOLOUR
    swc_compare = np.vstack((swc1, swc2_fn))
    swc_compare[:, -2]  = 1

    # Compute the SD, SSD, SSD% defined in Peng.et.al 2010
    SD = (np.mean(mindist1) + np.mean(mindist2)) / 2
    far1, far2 = mindist1[mindist1 > dist1], mindist2[mindist2 > dist2]
    SSD = (np.mean(far1) + np.mean(far2)) / 2
    pSSD = (len(far1) / len(mindist1) + len(far2) / len(mindist2)) / 2

    return (precision, recall, f1), (SD, SSD, pSSD), swc_compare


tracer = R2Tracer(quality=False, silent=True, speed=False, clean=False, non_stop=False)
prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_full'
# following swc is cropped
gtswcprefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop'
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
        predswc, soma = tracer.trace(img=pred_im, threshold=8)
        swcarray = predswc.get_array()
        #swc N x 7 0 sampleid 1 typeid 2 xloc 3 yloc 4 zloc 5 radius 6 parentid
        predswc.save(fname=filepath+'.swc')
        filenum = file.split('.')[0].split('_')[0]
        print(filenum)
        gtswcpath = os.path.join(gtswcprefix,filenum+'.swc')
        gtswc = loadswc(filepath=gtswcpath)
        (precision, recall, f1), (SD, SSD, pSSD), swc_compare = precision_recall(swc1=swcarray,
                                                                                 swc2=gtswc,
                                                                                 dist1=4,
                                                                                 dist2=4)
        print(precision, recall, f1)




