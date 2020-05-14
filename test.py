"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util

# for hardcode test
import numpy as np


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


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.tif'):
                path = os.path.join(root, fname)
                images.append(path)

    # print(len(images[:min(max_dataset_size, len(images))]))
    return images[:min(max_dataset_size, len(images))]


# just for 3d sliding window design
def sort_slide_window(original, crop_size):
    a = original.copy()
    x, y, z = a.shape
    crop_x, crop_y, crop_z = crop_size
    pad_x = crop_size[0] - x % crop_size[0]
    pad_y = crop_size[1] - y % crop_size[1]
    pad_z = crop_size[2] - z % crop_size[2]
    pad_range = [pad_x, pad_y, pad_z]
    slice_windows = np.array([])
    slice_range = np.array([])
    slice_no = [int((x + pad_x) / crop_x), int((y + pad_y) / crop_y), int((z + pad_z) / crop_z)]
    padded_image = np.pad(a, ((0, pad_x), (0, pad_y), (0, pad_z)), 'constant')
    # writetiff3d(os.path.join(input_path, '1_padded.tif'),a)

    for n_x in range(int((x + pad_x) / crop_x)):
        for n_y in range(int((y + pad_y) / crop_y)):
            for n_z in range(int((z + pad_z) / crop_z)):
                if slice_windows.size == 0:
                    slice_windows = np.array([padded_image[crop_x * n_x:crop_x * (n_x + 1), crop_y * n_y:crop_y * (n_y + 1),
                                              crop_z * n_z:crop_z * (n_z + 1)]])
                    slice_range = np.array([[crop_x * n_x, crop_x * (n_x + 1), crop_y * n_y, crop_y * (n_y + 1),
                                             crop_z * n_z, crop_z * (n_z + 1)]])
                else:
                    # print(slice_windows.shape)
                    # print(np.array(a[crop_x * n_x:crop_x * (n_x + 1), crop_y * n_y:crop_y * (n_y + 1),
                    #     crop_z * n_z:crop_z * (n_z + 1)]).shape)
                    slice_windows = np.vstack((slice_windows, [
                        padded_image[crop_x * n_x:crop_x * (n_x + 1), crop_y * n_y:crop_y * (n_y + 1),
                        crop_z * n_z:crop_z * (n_z + 1)]]))
                    slice_range = np.vstack((slice_range, [
                        [crop_x * n_x, crop_x * (n_x + 1), crop_y * n_y, crop_y * (n_y + 1), crop_z * n_z,
                         crop_z * (n_z + 1)]]))
    #             a[crop_x*n_x:crop_x*(n_x+1), crop_y*n_y:crop_y*(n_y+1), crop_z*n_z:crop_z*(n_z+1)]

    return padded_image, pad_range, slice_windows, slice_range, slice_no

# def concate():
#     print(slice_windows.shape)
#     concate_result = np.zeros((slice_no[0] * crop_x, slice_no[1] * crop_y, slice_no[2] * crop_z))
#     for i in range(slice_no[0] * slice_no[1] * slice_no[2]):
#         print(slice_range[i])
#         print(slice_windows[i].shape)
#         x_s, x_e, y_s, y_e, z_s, z_e = slice_range[i]
#         print(concate_result[x_s:x_e, y_s:y_e, z_s:z_e].shape)
#         concate_result[x_s:x_e, y_s:y_e, z_s:z_e] = slice_windows[i]
#
#     remove_pad = concate_result[:slice_no[0] * crop_x - pad_x, :slice_no[1] * crop_y - pad_y,
#                  :slice_no[2] * crop_z - pad_z]
#     print(remove_pad.shape)
#     writetiff3d(os.path.join(input_path, '1_padded_reconstruct.tif'), a)
#
#     print(np.sum(remove_pad - original))



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if opt.save_type == '2d':
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            if opt.save_type == '2d':
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            # elif opt.save_type == '3d':
            #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, save_type='3d')
    elif opt.save_type == '3d':
            dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            # dir_A = os.path.join(opt.dataroot, 'taiwan_gt')  # get the image directory
            # dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            A_paths = make_dataset(dir_A, opt.max_dataset_size)
            # B_paths = make_dataset(dir_B, opt.max_dataset_size)
            A_paths.sort(key=lambda x: int(x.rstrip(".tif").split("/")[-1]))
            # B_paths.sort(key=lambda x: int(x.rstrip(".tif").split("/")[-1]))
            for path in A_paths:
                id = path.split('/')[-1].split('_')[0]
                id = id.split('.')[0]
                real_A = loadtiff3d(path)
                real_A = real_A.astype(float) / 255.0
                # real_B = real_B.astype(float) / 255.0
                crop_size = np.array(opt.crop_size_3d.split('x')).astype(np.int)
                padded_image, pad_range, slice_windows, slice_range, slice_no = sort_slide_window(real_A, crop_size)
                padded_fake_b = np.empty_like(padded_image)
                assert slice_windows.shape[0] == slice_no[0] * slice_no[1] * slice_no[2]
                for i in range(slice_no[0] * slice_no[1] * slice_no[2]):
                    # print(slice_range[i])
                    x_s, x_e, y_s, y_e, z_s, z_e = slice_range[i]
                    in_A = padded_image[x_s:x_e, y_s:y_e, z_s:z_e]
                    in_data = {'A': in_A, 'B': in_A, 'A_paths': None, 'B_paths': None}
                    model.set_input(in_data)
                    model.test()
                    fake_B = model.get_fake_B()
                    fake_B = fake_B.data[0].cpu().float().numpy()
                    # print(np.unique(fake_B))
                    fake_B[fake_B < 0] = 0
                    fake_B *= 255
                    # print(np.unique(fake_B))
                    fake_B[fake_B >= 255] = 255
                    # print(fake_B.shape)
                    padded_fake_b[x_s:x_e, y_s:y_e, z_s:z_e] = fake_B
                uppadded_fake_b = padded_fake_b[:slice_no[0] * crop_size[0] - pad_range[0],
                                  :slice_no[1] * crop_size[1] - pad_range[1],
                                  :slice_no[2] * crop_size[2] - pad_range[2]]
                # print(real_A.shape, uppadded_fake_b.shape)
                assert real_A.shape == uppadded_fake_b.shape
                result_path = os.path.join(opt.results_dir, opt.name, 'test_full')
                util.util.mkdir(result_path)
                util.util.save_image_3d(uppadded_fake_b, os.path.join(result_path, id + '_fake_B.tif'))
                # util.util.save_image_3d(loadtiff3d(path[1]), os.path.join(result_path, id + '_real_B.tif'))





    webpage.save()  # save the HTML
