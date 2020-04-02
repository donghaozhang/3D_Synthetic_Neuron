#!/usr/bin/env bash

python test.py --dataroot datasets/datasets/fly/fly3d/ --name 3d_unet_pixel --checkpoints_dir checkpoints --model pix2pix --netG unet_3d_cust --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir checkpoints --save_type 3d
