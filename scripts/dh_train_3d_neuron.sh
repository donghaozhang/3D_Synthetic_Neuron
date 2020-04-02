#!/usr/bin/env bash

python train.py --dataroot datasets/datasets/fly/fly3d --name 3d_unet_pixel --checkpoints_dir checkpoints --model pix2pix --netG unet_3d_cust --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d