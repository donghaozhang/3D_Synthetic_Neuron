set -ex

# 3D U-Net [DONE]
# G:unet D:pixel
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_unet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d_cust --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d
# python train.py --dataroot datasets/datasets/fly/fly3d --name 3d_unet_pixel --checkpoints_dir checkpoints --model pix2pix --netG unet_3d_cust --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d --no_html

# G:unet D:patch
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_unet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d_cust --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# 3D Link-Net
# G:linknet D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_linknet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:linknet D:patch [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_linknet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d


# 3D V-Net
# G:vnet D:pixel 【DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_vnet_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG vnet --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:vnet D:patch 【DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_vnet_patch --checkpoints_dir ../checkpoints --model pix2pix --netG vnet --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# 3D Student
# G:student D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_student_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG student --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:student D:patch [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_student_patch --checkpoints_dir ../checkpoints --model pix2pix --netG student --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# deeper student
# G:student D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_deeper_student_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG deeper_student --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:student D:patch
python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_deeper_student_patch --checkpoints_dir ../checkpoints --model pix2pix --netG deeper_student --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# unet_mrf
# G:MRF D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_mrf_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# unet_mrf_v2
# G:MRF2 D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_mrf_v2_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student_v2 --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:MRF2 D:patch [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_mrf_v2_patch --checkpoints_dir ../checkpoints --model pix2pix --netG moreinception_student_v2 --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d


# 3D ESP-Net
# G:esp D:pixel [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_esp_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG espnet --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:esp D:patch [DONE]
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_esp_patch --checkpoints_dir ../checkpoints --model pix2pix --netG espnet --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# 3D FCN-Net [BOOM SHAKALAKA]
# G:fcn D:pixel
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_fcn_pixel --checkpoints_dir ../checkpoints --model pix2pix --netG 3dfcn --gan_mode vanilla --netD pixel_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d

# G:fcn D:patch
# python ../train.py --dataroot ../datasets/datasets/fly/fly3d --name 3d_fcn_patch --checkpoints_dir ../checkpoints --model pix2pix --netG 3dfcn --gan_mode vanilla --netD basic_3d --direction AtoB --lambda_L1 100 --dataset_mode neuron3d --crop_size_3d 128x128x32 --input_nc 1 --output_nc 1 --norm batch --pool_size 0 --batch_size 2 --save_type 3d