import os.path
from data.base_dataset import BaseDataset, get_params, get_params_3d, get_transform, get_transform_3d, get_coverage_3d
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import cv2


class Neuron3DDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase+'A')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase+'B')
        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)
        # print(self.A_paths)
        self.A_paths.sort(key=lambda x: int(x.rstrip("_gt.tif").split("/")[-1]))
        # print(self.A_paths[0].rstrip("_gt.tif").rstrip("_gt.tif"))
        self.B_paths.sort(key=lambda x: int(x.rstrip(".tif").split("/")[-1]))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # A is GT (0-255), B is neuro image (0-255)
        A = self.loadtiff3d(A_path)
        B = self.loadtiff3d(B_path)
        A = A.astype(float) / 255.0
        B = B.astype(float) / 255.0

        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # transform_params = get_params_3d(self.opt, A.shape)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # size: NCHW
        A, B = get_coverage_3d(self.opt.crop_size_3d, A, B)
        # TODO: Add augumentation?

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def loadtiff3d(self, filepath):
        """Load a tiff file into 3D numpy array"""

        import tifffile as tiff
        a = tiff.imread(filepath)

        stack = []
        for sample in a:
            stack.append(np.rot90(np.fliplr(np.flipud(sample))))
        out = np.dstack(stack)

        return out

