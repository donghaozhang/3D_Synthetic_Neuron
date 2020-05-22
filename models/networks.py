import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

from models.util import *
import math


###############################################################################
# Networks
###############################################################################
class unet3d(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True,
    ):
        super(unet3d, self).__init__()
        # print('unet3d is being called')
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2_3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2_3d(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        # self.conv4 = unetConv2_3d(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2_3d(filters[3], filters[4], self.is_batchnorm)
        self.center = unetConv2_3d(filters[2], filters[3], self.is_batchnorm)

        # upsampling
        # self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        # self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('conv1', conv1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print('conv2', conv2.size())

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print('conv3', conv3.size())

        center = self.center(maxpool3)
        # up4 = self.up_concat4(conv3, center)
        up3 = self.up_concat3(conv3, center)
        # print('up3', up3.size())
        up2 = self.up_concat2(conv2, up3)
        # print('up2', up2.size())
        up1 = self.up_concat1(conv1, up2)
        # print('up1', up1.size())
        final = self.final(up1)
        # print('final:', final.size())

        return final


class unet3d_fea_extractor(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True,
    ):
        super(unet3d_fea_extractor, self).__init__()
        # print('unet3d is being called')
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2_3d(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2_3d(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        # self.conv4 = unetConv2_3d(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2_3d(filters[3], filters[4], self.is_batchnorm)
        self.center = unetConv2_3d(filters[2], filters[3], self.is_batchnorm)

        # upsampling
        # self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        # self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print('conv1', conv1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print('conv2', conv2.size())

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print('conv3', conv3.size())

        center = self.center(maxpool3)
        # up4 = self.up_concat4(conv3, center)
        up3 = self.up_concat3(conv3, center)
        # print('up3', up3.size())
        up2 = self.up_concat2(conv2, up3)
        # print('up2', up2.size())
        up1 = self.up_concat1(conv1, up2)
        # print('up1', up1.size())
        final = self.final(up1)
        # print('final:', final.size())

        return final, up3, up2, up1


class unet3dregStudentRes(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=1,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True,
    ):
        super(unet3dregStudentRes, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256] # 16, 32, 64
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm, residual_path=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
         # 1x1 convolutions are used to compute reductions before the expensive 3x3 convolutions
        self.conv_mid = nn.Conv3d(filters[0], filters[1], kernel_size=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = unetConv2_3d_regression(filters[1], filters[2], self.is_batchnorm, residual_path=True)

        # upsampling
        self.up_concat2 = unetUp3d_regression(filters[2], filters[1], self.is_deconv, residual_path=True)
        self.up_concat1 = unetUp3d_regression(filters[1], filters[0], self.is_deconv, residual_path=True)

        # final conv (without any concat)
        self.smartFinal = nn.Conv3d(filters[0], n_classes, 1)

        self.smartTanh = nn.Tanh()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv_mid = self.conv_mid(maxpool1)
        maxpool2 = self.maxpool2(conv_mid)

        conv3 = self.conv3(maxpool2)

        up2 = self.up_concat2(conv_mid, conv3)
        up1 = self.up_concat1(conv1, up2)
        final = self.smartFinal(up1)
        final = self.smartTanh(final)
        return final


class DeeperStudentRes(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=1,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True,
    ):
        super(DeeperStudentRes, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512] # 16, 32, 64
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm, residual_path=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        # 1x1 convolutions are used to compute reductions before the expensive 3x3 convolutions
        self.conv_mid = nn.Conv3d(filters[0], filters[1], kernel_size=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = unetConv2_3d_regression(filters[1], filters[2], self.is_batchnorm, residual_path=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = unetConv2_3d_regression(filters[2], filters[3], self.is_batchnorm, residual_path=True)

        # upsampling
        self.up_concat3 = unetUp3d_regression(filters[3], filters[2], self.is_deconv, residual_path=True)
        self.up_concat2 = unetUp3d_regression(filters[2], filters[1], self.is_deconv, residual_path=True)
        self.up_concat1 = unetUp3d_regression(filters[1], filters[0], self.is_deconv, residual_path=True)

        # final conv (without any concat)
        self.smartFinal = nn.Conv3d(filters[0], n_classes, 1)

        self.smartTanh = nn.Tanh()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv_2 = self.conv_mid(maxpool1)
        maxpool2 = self.maxpool2(conv_2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)

        up3 = self.up_concat3(conv3, conv4)
        up2 = self.up_concat2(conv_2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.smartFinal(up1)
        final = self.smartTanh(final)
        return final


class unet_mrf(nn.Module):
    def __init__(
            self,
            feature_scale=4,
            n_classes=1,
            is_deconv=True,
            in_channels=1,
            is_batchnorm=True,
    ):
        super(unet_mrf, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # downsampling
        filters = [64, 128, 256, 512]  # 16, 32, 64
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm, residual_path=True)
        self.conv_mid = MRF(filters[0], filters[1])
        self.conv3 = MRF(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = unetConv2_3d_regression(filters[2], filters[3], self.is_batchnorm, residual_path=True)

        # upsampling
        self.up_concat3 = unetUp3d_regression(filters[3], filters[2], self.is_deconv, residual_path=True)
        self.up_concat2 = unetUp3d_regression(filters[2], filters[1], self.is_deconv, residual_path=True)
        self.up_concat1 = unetUp3d_regression(filters[1], filters[0], self.is_deconv, residual_path=True)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        # print('conv1', conv1.size())
        # maxpool1 = self.maxpool1(conv1)
        # print('maxpool1', maxpool1.size())

        conv_2 = self.conv_mid(conv1)
        # print('conv2', conv_2.size())
        # maxpool2 = self.maxpool2(conv_2)
        # print('maxpool2', maxpool2.size())

        conv3 = self.conv3(conv_2)
        # print('conv3', conv3.size())
        maxpool3 = self.maxpool3(conv3)
        # print('maxpool3', maxpool3.size())

        conv4 = self.conv4(maxpool3)
        # print('conv4', conv4.size())

        up3 = self.up_concat3(conv3, conv4)
        up2 = self.up_concat2(conv_2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        final = self.tanh(final)
        # print('final', final.size())
        return final


class unet_mrf_v2(nn.Module):
    def __init__(
            self,
            feature_scale=4,
            n_classes=1,
            is_deconv=True,
            in_channels=1,
            is_batchnorm=True,
    ):
        super(unet_mrf_v2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # downsampling
        filters = [64, 128, 256, 512]  # 16, 32, 64
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm, residual_path=True)
        self.conv_mid = MRF(filters[0], filters[1])
        self.conv3 = MRF(filters[1], filters[2])
        self.conv4 = MRF(filters[2], filters[3])

        # upsampling
        self.up_concat3 = unetUp3d_regression(filters[3], filters[2], self.is_deconv, residual_path=True)
        self.up_concat2 = unetUp3d_regression(filters[2], filters[1], self.is_deconv, residual_path=True)
        self.up_concat1 = unetUp3d_regression(filters[1], filters[0], self.is_deconv, residual_path=True)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv_2 = self.conv_mid(conv1)
        conv3 = self.conv3(conv_2)
        conv4 = self.conv4(conv3)

        up3 = self.up_concat3(conv3, conv4)
        up2 = self.up_concat2(conv_2, up3)
        up1 = self.up_concat1(conv1, up2)
        final = self.final(up1)
        final = self.tanh(final)
        # print('final', final.size())
        return final


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        # print(x.size())
        out16 = self.in_tr(x)
        # print('out16: ', out16.size())
        out32 = self.down_tr32(out16)
        # print('out32: ', out32.size())
        out64 = self.down_tr64(out32)
        # print('out64: ', out64.size())
        out128 = self.down_tr128(out64)
        # print('out128: ', out128.size())
        out256 = self.down_tr256(out128)
        # print('out256: ', out256.size())
        out = self.up_tr256(out256, out128)
        # print('up256: ', out.size())
        out = self.up_tr128(out, out64)
        # print('up128: ', out.size())
        out = self.up_tr64(out, out32)
        # print('up64: ', out.size())
        out = self.up_tr32(out, out16)
        # print('up32: ', out.size())
        out = self.out_tr(out)
        # print('final: ', out.size())
        return out


class linknet3D(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=1, is_batchnorm=True, layers=18
    ):
        super(linknet3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        if layers == 18:
            self.layers = [2, 2, 2, 2] # Currently hardcoded for ResNet-18
        elif layers == 34:
            self.layers = [3, 4, 6, 3]

        filters = [64, 128, 256, 512]
        # filters = [int(x / self.feature_scale) for x in filters]

        self.inplanes = filters[0]

        # Encoder
        self.convbnrelu1 = conv3DBatchNormRelu(
            in_channels=1, k_size=7, n_filters=64, padding=3, stride=2, bias=False
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        block = residualBlock3D
        self.encoder1 = self._make_layer(block, filters[0], self.layers[0])
        self.encoder2 = self._make_layer(block, filters[1], self.layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], self.layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], self.layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(7)

        # Decoder
        self.decoder4 = linknetUp3D(filters[3], filters[2], 3, 2, 1, 1)
        self.decoder3 = linknetUp3D(filters[2], filters[1], 3, 2, 1, 1)
        self.decoder2 = linknetUp3D(filters[1], filters[0], 3, 2, 1, 1)
        self.decoder1 = linknetUp3D(filters[0], filters[0], 3, 1, 1, 0)

        # Final Classifier
        self.finaldeconvbnrelu1 = nn.Sequential(
            nn.ConvTranspose3d(filters[0], 32, 3, 2, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.finalconvbnrelu2 = conv3DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=1,
        )
        self.finalconv3 = nn.ConvTranspose3d(32, n_classes, 2, 2, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        # print('x', x.shape)
        x = self.convbnrelu1(x)
        # print('x', x.shape)
        x = self.maxpool(x)
        # print('x', x.shape)

        e1 = self.encoder1(x)
        # print('e1', e1.shape)
        e2 = self.encoder2(e1)
        # print('e2', e2.shape)
        e3 = self.encoder3(e2)
        # print('e3', e3.shape)
        e4 = self.encoder4(e3)
        # print('e4', e4.shape)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        # print('d4', d4.shape)
        # d4 += e3
        # print('d4', d4.shape)
        d3 = self.decoder3(d4) + e2
        # print('d3', d3.shape)
        # d3 += e2
        # print('d3', d3.shape)
        d2 = self.decoder2(d3) + e1
        # print('d2', d2.shape)
        # d2 += e1
        # print('d2', d2.shape)
        d1 = self.decoder1(d2)
        # print('d1', d1.shape)

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        # print('f1', f1.shape)
        f2 = self.finalconvbnrelu2(f1)
        # print('f2', f2.shape)
        f3 = self.finalconv3(f2)
        # print('f3', f3.shape)

        return f3


class ESPNet(nn.Module):
    def __init__(self, classes=1, channels=1):
        super().__init__()
        self.input1 = InputProjectionA(1)
        self.input2 = InputProjectionA(1)

        initial = 16 # feature maps at level 1
        config = [32, 128, 256, 256] # feature maps at level 2 and onwards
        reps = [2, 2, 3]

        ### ENCODER

        # all dimensions are listed with respect to an input  of size 4 x 128 x 128 x 128
        self.level0 = CBR(channels, initial, 7, 2) # initial x 64 x 64 x64
        self.level1 = nn.ModuleList()
        for i in range(reps[0]):
            if i==0:
                self.level1.append(DilatedParllelResidualBlockB1(initial, config[0]))  # config[0] x 64 x 64 x64
            else:
                self.level1.append(DilatedParllelResidualBlockB1(config[0], config[0]))  # config[0] x 64 x 64 x64

        # downsample the feature maps
        self.level2 = DilatedParllelResidualBlockB1(config[0], config[1], stride=2) # config[1] x 32 x 32 x 32
        self.level_2 = nn.ModuleList()
        for i in range(0, reps[1]):
            self.level_2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        # downsample the feature maps
        self.level3_0 = DilatedParllelResidualBlockB1(config[1], config[2], stride=2) # config[2] x 16 x 16 x 16
        self.level_3 = nn.ModuleList()
        for i in range(0, reps[2]):
            self.level_3.append(DilatedParllelResidualBlockB1(config[2], config[2])) # config[2] x 16 x 16 x 16


        ### DECODER

        # upsample the feature maps
        self.up_l3_l2 = UpSampler(config[2], config[1])  # config[1] x 32 x 32 x 32
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l2 = DilatedParllelResidualBlockB1(2 * config[1], config[1]) # config[1] x 32 x 32 x 32
        self.dec_l2 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        self.up_l2_l1 = UpSampler(config[1], config[0])  # config[0] x 64 x 64 x 64
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l1 = DilatedParllelResidualBlockB1(2*config[0], config[0]) # config[0] x 64 x 64 x 64
        self.dec_l1 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l1.append(DilatedParllelResidualBlockB1(config[0], config[0])) # config[0] x 64 x 64 x 64

        self.dec_l1.append(CBR(config[0], classes, 3, 1)) # classes x 64 x 64 x 64
        # We use ESP block without reduction step because the number  of input feature maps are very small (i.e. 4 in
        # our case)
        self.dec_l1.append(ASPBlock(classes, classes))

        # Using PSP module to learn the representations at different scales
        self.pspModules = nn.ModuleList()
        scales = [0.2, 0.4, 0.6, 0.8]
        for sc in scales:
             self.pspModules.append(PSPDec(classes, classes, sc))

        # Classifier
        self.classifier = self.classifier = nn.Sequential(
             CBR((len(scales) + 1) * classes, classes, 3, 1),
             ASPBlock(classes, classes), # classes x 64 x 64 x 64
             nn.Upsample(scale_factor=2), # classes x 128 x 128 x 128
             CBR(classes, classes, 7, 1), # classes x 128 x 128 x 128
             C(classes, classes, 1, 1) # classes x 128 x 128 x 128
        )
        #

        for m in self.modules():
             if isinstance(m, nn.Conv3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             if isinstance(m, nn.ConvTranspose3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             elif isinstance(m, nn.BatchNorm3d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()

    def forward(self, input1, inp_res=(128, 128, 128), inpSt2=False):
        dim0 = input1.size(2)
        dim1 = input1.size(3)
        dim2 = input1.size(4)

        if self.training or inp_res is None:
            # input resolution should be divisible by 8
            inp_res = (math.ceil(dim0 / 8) * 8, math.ceil(dim1 / 8) * 8,
                       math.ceil(dim2 / 8) * 8)
        if inp_res:
            input1 = F.adaptive_avg_pool3d(input1, output_size=inp_res)

        out_l0 = self.level0(input1)

        for i, layer in enumerate(self.level1): #64
            if i == 0:
                out_l1 = layer(out_l0)
            else:
                out_l1 = layer(out_l1)

        out_l2_down = self.level2(out_l1) #32
        for i, layer in enumerate(self.level_2):
            if i == 0:
                out_l2 = layer(out_l2_down)
            else:
                out_l2 = layer(out_l2)
        del out_l2_down

        out_l3_down = self.level3_0(out_l2) #16
        for i, layer in enumerate(self.level_3):
            if i == 0:
                out_l3 = layer(out_l3_down)
            else:
                out_l3 = layer(out_l3)
        del out_l3_down

        dec_l3_l2 = self.up_l3_l2(out_l3)
        merge_l2 = self.merge_l2(torch.cat([dec_l3_l2, out_l2], 1))
        for i, layer in enumerate(self.dec_l2):
            if i == 0:
                dec_l2 = layer(merge_l2)
            else:
                dec_l2 = layer(dec_l2)

        dec_l2_l1 = self.up_l2_l1(dec_l2)
        merge_l1 = self.merge_l1(torch.cat([dec_l2_l1, out_l1], 1))
        for i, layer in enumerate(self.dec_l1):
            if i == 0:
                dec_l1 = layer(merge_l1)
            else:
                dec_l1 = layer(dec_l1)

        psp_outs = dec_l1.clone()
        for layer in self.pspModules:
            out_psp = layer(dec_l1)
            psp_outs = torch.cat([psp_outs, out_psp], 1)

        decoded = self.classifier(psp_outs)
        return F.upsample(decoded, size=(dim0, dim1, dim2), mode='trilinear')


class fcn3dnet(nn.Module):
    def __init__(self, n_classes=1):
        super(fcn3dnet, self).__init__()

        # stem
        self.block1 = stem(conv_in_channels=1)

        # deconvolution 2x
        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=7, stride=2),
                                     nn.ConvTranspose3d(in_channels=96, out_channels=1, kernel_size=(6,6,2)))

        # inceptionA
        # self.block2 = inceptionA(conv_in_channels=192)
        self.block2 = inceptionA(conv_in_channels=192)

        # deconvolution 4x
        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=384, out_channels=192, kernel_size=(3,3,1), stride=2,
                                          output_padding=0),
                                     nn.ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=7, stride=2),
                                     nn.ConvTranspose3d(in_channels=96, out_channels=1, kernel_size=(6, 6, 2)))

        # reductionA
        self.block3 = reductionA(conv_in_channels=384)

        # inceptionB
        self.block4 = inceptionB(conv_in_channels=896)

        # deconvolution 8x
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=896, out_channels=384, kernel_size=(3, 3, 1), stride=2),
            nn.ConvTranspose3d(in_channels=384, out_channels=192, kernel_size=(3, 3, 1), stride=2,
                               output_padding=0),
            nn.ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=7, stride=2),
            nn.ConvTranspose3d(in_channels=96, out_channels=1, kernel_size=(6, 6, 2)))

        # reductionB
        self.block5 = reductionB(conv_in_channels=896)

        # inceptionC
        self.block6 = inceptionC(conv_in_channels=2048)

        # deconvolution 8x
        # self.deconv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=2048, out_channels=896, kernel_size=(3, 3, 1), stride=2, output_padding=(1,1,0)),
        #                              nn.ConvTranspose3d(in_channels=896, out_channels=384, kernel_size=(3, 3, 1),
        #                                                 stride=2),
        #                              nn.ConvTranspose3d(in_channels=384, out_channels=192, kernel_size=(3, 3, 1),
        #                                                 stride=2,
        #                                                 output_padding=0),
        #                              nn.ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=7, stride=2),
        #                              nn.ConvTranspose3d(in_channels=96, out_channels=1, kernel_size=(6, 6, 2)))
        self.deconv4_1 = nn.ConvTranspose3d(in_channels=2048, out_channels=896, kernel_size=(3, 3, 1), stride=2, output_padding=(1,1,1))
        self.deconv4_2 = nn.ConvTranspose3d(in_channels=896, out_channels=384, kernel_size=(3, 3, 1), stride=2)
        self.deconv4_3 = nn.ConvTranspose3d(in_channels=384, out_channels=192, kernel_size=(3, 3, 1), stride=2, output_padding=0)
        self.deconv4_4 = nn.ConvTranspose3d(in_channels=192, out_channels=96, kernel_size=7, stride=2)
        self.deconv4_5 = nn.ConvTranspose3d(in_channels=96, out_channels=1, kernel_size=(6, 6, 2))

        #todo

    def forward(self,x):
        print('The input size is: ' + str(x.size()))
        out = self.block1(x)
        print('The size after stem: ' + str(out.size()))
        out_deconv1 = self.deconv1(out)
        print('The size after deconv1: ' + str(out_deconv1.size()))
        out = self.block2(out)
        print('The size after inceptionA: ' + str(out.size()))
        out_deconv2 = self.deconv2(out)
        print('The size after deconv2: ' + str(out_deconv2.size()))
        out = self.block3(out)
        print('The size after reductionA: ' + str(out.size()))
        out = self.block4(out)
        print('The size after inceptionB: ' + str(out.size()))
        out_deconv3 = self.deconv3(out)
        print('The size after deconv3: ' + str(out_deconv3.size()))
        out = self.block5(out)
        print('The size after reductionB: ' + str(out.size()))
        out = self.block6(out)
        print('The size after inceptionC: ' + str(out.size()))
        # out_deconv4 = self.deconv4(out)
        # print('The size after deconv4: ' + str(out_deconv4.size()))

        out_deconv4_1 = self.deconv4_1(out)
        print('size deconv4_1: ', out_deconv4_1.size())
        out_deconv4_2 = self.deconv4_2(out_deconv4_1)
        print('size deconv4_2: ', out_deconv4_2.size())
        out_deconv4_3 = self.deconv4_3(out_deconv4_2)
        print('size deconv4_3: ', out_deconv4_3.size())
        out_deconv4_4 = self.deconv4_4(out_deconv4_3)
        print('size deconv4_4: ', out_deconv4_4.size())
        out_deconv4_5 = self.deconv4_5(out_deconv4_4)
        print('size deconv4_5: ', out_deconv4_5.size())

        add_all_deconv = torch.add(out_deconv1, out_deconv2)
        add_all_deconv = torch.add(add_all_deconv, out_deconv3)
        add_all_deconv = torch.add(add_all_deconv, out_deconv4_5)

        return add_all_deconv

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_3d':
        net = Unet3DGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_3d_cust':
        net = unet3d()
    elif netG == 'linknet_3d':
        net = linknet3D()
    elif netG == 'vnet':
        net = VNet()
    elif netG == 'student':
        net = unet3dregStudentRes()
    elif netG == 'deeper_student':
        net = DeeperStudentRes()
    elif netG == 'unet_mrf':
        net = unet_mrf()
    elif netG == 'unet_mrf_v2':
        net = unet_mrf_v2()
    elif netG == 'espnet':
        net = ESPNet()
    elif netG == '3dfcn':
        net = fcn3dnet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_3d':
        net = NLayer3D_Discriminator(2, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'pixel_3d':
        net = PixelDiscriminator3D(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Unet3DGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet3DGenerator, self).__init__()
        # construct unet structure
        unet_block = Unet3DSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = Unet3DSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = Unet3DSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = Unet3DSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = Unet3DSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = Unet3DSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Unet3DSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(Unet3DSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm3d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm3d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayer3D_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayer3D_Discriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm3d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm3d

        use_bias = False
        kw = 4

        self.conv1 = nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, True)
        nf_mult = 1
        nf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence += [
        #         nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=1, bias=use_bias),
        #         nn.BatchNorm3d(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]
        #
        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=1, bias=use_bias),
        #     nn.BatchNorm3d(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]

        self.conv2 = conv3DBatchNormLeakyRelu(ndf, ndf*2, k_size=kw, stride=2, padding=1)
        # self.conv3 = conv3DBatchNormLeakyRelu(ndf*2, ndf * 4, k_size=kw, stride=2, padding=1)
        # self.conv4 = conv3DBatchNormLeakyRelu(ndf * 4, ndf * 8, k_size=kw, stride=2, padding=1)

        self.final = nn.Conv3d(ndf * 2, 1, kernel_size=kw, stride=1, padding=1)  # output 1 channel prediction map

    def forward(self, input):
        """Standard forward."""
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)
        conv2 = self.conv2(relu1)
        # conv3 = self.conv3(conv2)
        # conv4 = self.conv4(conv3)
        final = self.final(conv2)
        return final


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PixelDiscriminator3D(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
