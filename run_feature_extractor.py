#python run_feature_extractor.py --dataroot 'datasets/datasets/fly/fly3d/'
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from torch.autograd import Variable
from models.networks import unet3d, unet3d_fea_extractor
if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.name = '3d_unet_pixel'
    opt.checkpoints_dir = 'checkpoints'
    opt.model = 'pix2pix'
    opt.netG = 'unet_3d_cust'
    opt.direction = 'AtoB'
    opt.dataset_mode = 'neuron3d'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.crop_size_3d = '128x128x32'
    opt.norm = 'batch'
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    # print(model.netG)

    netG = unet3d_fea_extractor()
    epoch = 'latest'
    name = 'G'
    load_filename = '%s_net_%s.pth' % (epoch, name)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    load_path = os.path.join(save_dir, load_filename)
    modelstate = torch.load(load_path)
    netG.load_state_dict(modelstate)

    #
    input = torch.FloatTensor(1, 1, 128, 64, 64)
    input_var = Variable(input).cuda()
    netG.eval()
    netG(input)
    # mask = torch.FloatTensor(bSz, channels, 256, 256)
    # mask_var = Variable(mask).cuda()
    # model = SAUNet_gate().eval().cuda()
    # out = model(input_var)