import numpy as np
import skfmm
import os


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


def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out


def writetiff3d(filepath, block):
    import tifffile as tiff

    try:
        os.remove(filepath)
    except OSError:
        pass

    with tiff.TiffWriter(filepath, bigtiff=False) as tif:
        for z in range(block.shape[2]):
            saved_block = np.rot90(block[:, :, z])
            tif.save(saved_block.astype('uint8'), compress=0)


def crop(img):
    ind = np.argwhere(img > 0)
    x = ind[:, 0]
    y = ind[:, 1]
    z = ind[:, 2]
    xmin = max(x.min() - 5, 0)
    xmax = min(x.max() + 5, img.shape[0])
    ymin = max(y.min() - 5, 0)
    ymax = max(y.min() + 5, img.shape[1])
    zmin = max(z.min() - 5, 0)
    zmax = max(z.min() + 5, img.shape[2])

    return np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])



'''
Regenerate tif file from swc and also apply distance transform.
'''
def swc2tif_dt(swc, img):
    import math

    shape = img.shape
    skimg = np.ones(shape)
    zeromask = np.ones(shape)

    # Add nodes the current swc to make sure there is
    # at least one node in each voxel on a branch
    idlist = swc[:, 0]
    extra_nodes = []
    extra_nodes_radius = []
    for i in range(swc.shape[0]):
        cnode = swc[i, 2:5]
        c_radius = swc[i, -2]
        pnode = swc[idlist == swc[i, 6], 2:5]
        if pnode.shape[0] != 0:
            p_radius = swc[idlist == swc[i, 6], -2][0]
            average_radius = int(c_radius+p_radius)/2

        dvec = pnode - cnode # [[x, y, z]]
        dvox = np.floor(np.linalg.norm(dvec)) # eculidean norm
        if dvox >= 1:
            uvec = dvec / (dvox + 1) # unit vector
            extra_nodes.extend(
                [cnode + uvec * i for i in range(1, int(dvox))])
            extra_nodes_radius.extend([average_radius for i in range(1, int(dvox))])

    # Deal with nodes in swc
    for i in range(swc.shape[0]):
        node = [math.floor(n) for n in swc[i, 2:5]]
        for j in range(3):
            if node[j] > shape[j]-1:
                node[j] = shape[j]-1
        r = int(swc[i, -2])
        skimg[node[0], node[1], node[2]] = 0
        zeromask[max(0,node[0]-r): min(node[0]+r, shape[0]), max(0,node[1]-r):min(node[1]+r, shape[1]), max(0, node[2]-r):min(node[2]+r, shape[2])] = 0

    # Deal with the extra nodes
    ex_count = 0
    for ex in extra_nodes:
        node = [math.floor(n) for n in ex[0]] # get integer x, y, z
        for j in range(3):
            if node[j] > shape[j]-1:
                node[j] = shape[j]-1
        skimg[node[0], node[1], node[2]] = 0
        r = int(extra_nodes_radius[ex_count])
        zeromask[max(0,node[0]-r): min(node[0]+r, shape[0]), max(0,node[1]-r):min(node[1]+r, shape[1]), max(0, node[2]-r):min(node[2]+r, shape[2])] = 0
        ex_count += 1

    a, dm = 6, 5
    dt = skfmm.distance(skimg, dx=1)

    dt = np.exp(a * (1 - dt / dm)) - 1
    dt[zeromask == 1] = 0
    dt = (dt/np.max(dt))*255
    return dt


# Generate distance transform based on original image
# Store all 3D tif files and swc files into fly_original folder
def store_swc_tif():
    path_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/fly_original/'
    for i in range(1,43):
        img = loadtiff3d(path_prefix+str(i)+'.tif')
        swc = loadswc(path_prefix+str(i)+'.swc')
        img_gt = swc2tif_dt(swc,img)
        writetiff3d(path_prefix+str(i)+'_gt.tif', img_gt)


# Crop the image
def crop_im():
    import subprocess
    path_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/fly_original/'
    out_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop/'
    for i in range(1, 43):
        img = loadtiff3d(path_prefix + str(i) + '.tif')
        img_gt = loadtiff3d(path_prefix + str(i) + '_gt.tif')

        x, y, z = crop(img_gt)
        img_crop = img[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        img_gt = img_gt[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        writetiff3d(out_prefix + str(i) + '.tif', img_crop)
        writetiff3d(out_prefix + str(i) + '_gt.tif', img_gt)


# Split into training and testing
def split_train_test():
    import random
    ind = [x for x in range(43)]
    random.shuffle(ind)
    train = ind[:38]
    val = ind[38:]


def copy_file():
    import subprocess
    train = [12, 17, 31, 38, 42, 18, 26, 4, 13, 32, 15, 5, 34, 1, 23, 29, 9, 39, 3, 11, 27, 36, 41, 6, 28, 30, 19, 35, 21, 8, 2, 37, 14, 25, 7, 10, 24]
    val = [33, 16, 20, 22, 40]
    in_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop/'
    out_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/fly3d/'

    for t in train:
        subprocess.call(['cp', in_prefix + str(t) + '.tif', out_prefix + 'trainA/' + str(t) + '.tif'])
        subprocess.call(['cp', in_prefix + str(t) + '_gt.tif', out_prefix + 'trainB/' + str(t) + '_gt.tif'])

    for t in val:
        subprocess.call(['cp', in_prefix + str(t) + '.tif', out_prefix + 'testA/' + str(t) + '.tif'])
        subprocess.call(['cp', in_prefix + str(t) + '_gt.tif', out_prefix + 'testB/' + str(t) + '_gt.tif'])


# Crop the image
def crop_swc():
    import subprocess
    path_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/fly_original/'
    out_prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop/'
    for i in range(1, 43):
        img = loadtiff3d(path_prefix + str(i) + '.tif')
        img_gt = loadtiff3d(path_prefix + str(i) + '_gt.tif')
        swc = loadswc(path_prefix + str(i) + '.swc')
        x, y, z = crop(img_gt)
        swc[:,2] = swc[:,2]-x[0]
        swc[:,3] = swc[:,3]-y[0]
        swc[:,4] = swc[:,4]-z[0]
        swcoutpath = out_prefix + str(i) + '.swc'
        saveswc(swcoutpath, swc)


if __name__ == '__main__':
    crop_swc()