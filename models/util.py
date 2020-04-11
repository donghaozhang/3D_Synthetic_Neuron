import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


###############################################################################
# Basic Modules
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

class conv3DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv3DBatchNorm, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm3d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv3DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv3DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm3d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv3DBatchNormLeakyRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv3DBatchNormLeakyRelu, self).__init__()

        conv_mod = nn.Conv3d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm3d(int(n_filters)), nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.cbr_unit = nn.LeakyReLU(conv_mod, nn.ReLU(0.2, inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv3DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, output_padding=0, bias=True):
        super(deconv3DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(
            nn.ConvTranspose3d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                output_padding=output_padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm3d(int(n_filters)),
        )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class deconv3DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, output_padding=0, bias=True):
        super(deconv3DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(
            nn.ConvTranspose3d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                output_padding=output_padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm3d(int(n_filters)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class residualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock3D, self).__init__()

        self.convbnrelu1 = conv3DBatchNormRelu(in_channels, n_filters, 3, stride, 1, bias=False)
        self.convbn2 = conv3DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class linknetUp(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv3DBatchNormRelu(
            in_planes, in_planes/4, k_size=1, stride=1, padding=0
        )

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = deconv3DBatchNormRelu(
            in_planes/4, in_planes/4, k_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv3DBatchNormRelu(
            in_planes/4, out_planes, k_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # print(x.shape)
        x = self.convbnrelu1(x)
        # print(x.shape)
        x = self.deconvbnrelu2(x)
        # print(x.shape)
        x = self.convbnrelu3(x)
        # print(x.shape)
        return x


class unetConv2_3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2_3d, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm3d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm3d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, 3, 1, 1), nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetConv2_3d_regression(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, residual_path=False):
        super(unetConv2_3d_regression, self).__init__()
        self.residual_path = residual_path
        # print(self.residual_path)
        # if residual_path:
        #     self.reduce_channel = nn.Conv3d(in_size, in_size, 1)
        if residual_path:
            if is_batchnorm:
                self.conv1 = nn.Sequential(
                    nn.Conv3d(in_size, in_size, 3, 1, 1),
                    nn.BatchNorm3d(in_size),
                    nn.ReLU(),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv3d(in_size, out_size, 3, 1, 1),
                    nn.BatchNorm3d(out_size),
                    nn.ReLU(),
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size, 3, 1, 1), nn.ReLU())
                self.conv2 = nn.Sequential(
                    nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU()
                )
        else:
            if is_batchnorm:
                self.conv1 = nn.Sequential(
                    nn.Conv3d(in_size, out_size, 3, 1, 1),
                    nn.BatchNorm3d(out_size),
                    nn.ReLU(),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv3d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm3d(out_size),
                    nn.ReLU(),
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU())
                self.conv2 = nn.Sequential(
                    nn.Conv3d(out_size, out_size, 3, 1, 1), nn.ReLU()
                )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.residual_path:
            outputs = torch.add(inputs, outputs)
        outputs = self.conv2(outputs)


        return outputs


class unetUp3d_regression(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, residual_path=False):
        super(unetUp3d_regression, self).__init__()
        self.conv = unetConv2_3d_regression(in_size, out_size, False, residual_path=residual_path)
        if is_deconv:
            # self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
            self.up = nn.Sequential(nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2),
                                  nn.LeakyReLU(0.2, False))
        else:
            self.up = F.interpolate(scale_factor=2, mode='bilinear')


    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = outputs2.size()[2] - inputs1.size()[2]
        offset2 = outputs2.size()[3] - inputs1.size()[3]
        offset3 = outputs2.size()[4] - inputs1.size()[4]
        padding = [offset3 // 2, offset3 - offset3 // 2, offset2//2, offset2-offset2//2, offset1//2, offset1-offset1//2]
        outputs1 = F.pad(inputs1, padding)

        output = torch.cat([outputs1, outputs2], 1)

        output = self.conv(output)
        return output


# class inceptionA(nn.Module):
#     def __init__(self, conv_in_channels):
#         super(inceptionA, self).__init__()
#
#         self.layer1 = nn.ReLU(inplace=False)
#         self.layer2_1 = block(conv_in_channels, 192, (3,3,1), 2)
#
#         self.layer2_2 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=2)
#
#         self.layer3_1 = nn.Sequential(block(384, 32, 1),
#                                       block(32, 384, 1)
#                                       )
#
#
#         self.layer3_2 = nn.Sequential(block(384, 32, 1),
#                                       block(32, 32, (3, 3, 1), 1, (1, 1, 0)),
#                                       block(32, 384, 1))
#
#         self.layer3_3 = nn.Sequential(block(384, 32, 1),
#                                       block(32, 48, (3,3,1), 1, (1,1,0)),
#                                       block(48, 64, (3,3,1), 1, (1,1,0)),
#                                       block(64, 384, 1)
#                                       )
#
#     def forward(self, x):
#         # log('inceptionA - input: ' + str(x.size()))
#         relu = self.layer1(x)
#         # log('inceptionA - after layer1: ' + str(relu.size()))
#         conv1 = self.layer2_1(relu)
#         # log('inceptionA - after layer2_1: ' + str(conv1.size()))
#         conv2 = self.layer2_2(relu)
#         # log('inceptionA - after layer2_2: ' + str(conv2.size()))
#         concat = torch.cat((conv1, conv2), dim=1)
#         # log('inceptionA - after concat: ' + str(concat.size()))
#         path3_1 = self.layer3_1(concat)
#         # log('inceptionA - after layer3_1: ' + str(path3_1.size()))
#         path3_2 = self.layer3_2(concat)
#         # log('inceptionA - after layer3_2: ' + str(path3_2.size()))
#         path3_3 = self.layer3_3(concat)
#         # log('inceptionA - after layer3_3: ' + str(path3_3.size()))
#         out = torch.add(path3_1, path3_2)
#         # log('inceptionA - after concat first two: ' + str(out.size()))
#         out = torch.add(out, path3_3)
#         # log('inceptionA - after concat the 3rd one: ' + str(out.size()))
#         out = torch.add(out, concat)
#         # log('inceptionA - after concat the residual one: ' + str(out.size()))
#
#         # residual- add input after relu to the final output
#
#         return out


class MRF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRF, self).__init__()

        self.layer1 = conv3DBatchNormRelu(
            in_channels=in_channels,
            k_size=1,
            n_filters=out_channels,
            padding=0,
            stride=2,
        )

        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=2)
        self.layer2 = conv3DBatchNormRelu(
            in_channels=in_channels,
            k_size=1,
            n_filters=out_channels,
            padding=0,
            stride=1,
        )

        self.layer3_1 = conv3DBatchNormRelu(
            in_channels=in_channels,
            k_size=1,
            n_filters=out_channels,
            padding=0,
            stride=2,
        )

        self.layer3_2 = conv3DBatchNormRelu(
            in_channels=out_channels,
            k_size=3,
            n_filters=out_channels,
            padding=1,
            stride=1,
        )

        self.layer4_1 = conv3DBatchNormRelu(
            in_channels=in_channels,
            k_size=1,
            n_filters=out_channels,
            padding=0,
            stride=2,
        )

        self.layer4_2 = conv3DBatchNormRelu(
            in_channels=out_channels,
            k_size=3,
            n_filters=out_channels*2,
            padding=1,
            stride=1,
        )

        self.layer4_3 = conv3DBatchNormRelu(
            in_channels=out_channels*2,
            k_size=3,
            n_filters=out_channels,
            padding=1,
            stride=1,
        )

    def forward(self, x):
        # log('inceptionA - after concat: ' + str(concat.size()))
        path1_final = self.layer1(x)

        path2_1 = self.maxpool(x)
        path2_final = self.layer2(path2_1)

        path3_1 = self.layer3_1(x)
        path3_final = self.layer3_2(path3_1)

        path4_1 = self.layer4_1(x)
        path4_2 = self.layer4_2(path4_1)
        path4_final = self.layer4_3(path4_2)

        out = torch.add(path1_final, path2_final)
        out = torch.add(out, path3_final)
        out = torch.add(out, path4_final)

        return out


class unetUp3d(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp3d, self).__init__()
        self.conv = unetConv2_3d(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.functional.F.interpolate(scale_factor=2, mode='bilinear')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = outputs2.size()[2] - inputs1.size()[2]
        offset2 = outputs2.size()[3] - inputs1.size()[3]
        offset3 = outputs2.size()[4] - inputs1.size()[4]
        padding = [offset3 // 2, offset3 - offset3 // 2, offset2//2, offset2-offset2//2, offset1//2, offset1-offset1//2]
        outputs1 = nn.functional.pad(inputs1, padding)

        output = torch.cat([outputs1, outputs2], 1)

        output = self.conv(output)
        return output


class linknetUp3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0):
        super(linknetUp3D, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv3DBatchNormRelu(
            in_planes, in_planes/4, k_size=1, stride=1, padding=0
        )

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = deconv3DBatchNormRelu(
            in_planes/4, in_planes/4, k_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv3DBatchNormRelu(
            in_planes/4, out_planes, k_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # print(x.shape)
        x = self.convbnrelu1(x)
        # print(x.shape)
        x = self.deconvbnrelu2(x)
        # print(x.shape)
        x = self.convbnrelu3(x)
        # print(x.shape)
        return x


class residualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock3D, self).__init__()

        self.convbnrelu1 = conv3DBatchNormRelu(in_channels, n_filters, 3, stride, 1, bias=False)
        self.convbn2 = conv3DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


##############################################################################
# V-Net sub-modules
##############################################################################
def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        # self.bn1 = ContBatchNorm3d(nchan)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        # self.bn1 = ContBatchNorm3d(16)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        # print(x.size())
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)

        # print(out.size(), x16.size())
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        # self.bn1 = ContBatchNorm3d(outChans)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        # self.bn1 = ContBatchNorm3d(outChans // 2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        # self.bn1 = ContBatchNorm3d(2)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 1, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


##############################################################################
# ESP-Net sub-modules
##############################################################################
class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        k = 4
        n = int(nOut/k)
        n1 = nOut - (k-1)*n
        self.c1 = nn.Sequential(CBR(nIn, n, 1, 1), C(n, n, 3, 2))
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 3)
        self.d8 = CDilated(n, n, 3, 1, 4)
        self.bn = BR(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3],1)
        if input.size() == combine.size():
            combine = input + combine
        output = self.bn(combine)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        #self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        return self.conv(input)
        #return self.bn(output)


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool3d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        k = 4
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = CBR(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, stride, 1)
        self.d2 = CDilated(n, n, 3, stride, 1)
        self.d4 = CDilated(n, n, 3, stride, 2)
        self.d8 = CDilated(n, n, 3, stride, 2)
        self.bn = nn.BatchNorm3d(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = self.bn(torch.cat([d1, add1, add2, add3], 1))
        if input.size() == combine.size():
            combine = input + combine
        output = F.relu(combine, inplace=True)
        return output

class ASPBlock(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        self.d1 = CB(nIn, nOut, 3, 1)
        self.d2 = CB(nIn, nOut, 5, 1)
        self.d4 = CB(nIn, nOut, 7, 1)
        self.d8 = CB(nIn, nOut, 9, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        d1 = self.d1(input)
        d2 = self.d2(input)
        d3 = self.d4(input)
        d4 = self.d8(input)

        combine = d1 + d2 + d3 + d4
        if input.size() == combine.size():
            combine = input + combine
        output = self.act(combine)
        return output


class UpSampler(nn.Module):
    '''
    Up-sample the feature maps by 2
    '''
    def __init__(self, nIn, nOut):
        super().__init__()
        self.up = CBR(nIn, nOut, 3, 1)

    def forward(self, inp):
        return F.upsample(self.up(inp), mode='trilinear', scale_factor=2)


class PSPDec(nn.Module):
    '''
    Inspired or Adapted from Pyramid Scene Network paper
    '''

    def __init__(self, nIn, nOut, downSize):
        super().__init__()
        self.scale = downSize
        self.features = CBR(nIn, nOut, 3, 1)
    def forward(self, x):
        assert x.dim() == 5
        inp_size = x.size()
        out_dim1, out_dim2, out_dim3 = int(inp_size[2] * self.scale), int(inp_size[3] * self.scale), int(inp_size[4] * self.scale)
        x_down = F.adaptive_avg_pool3d(x, output_size=(out_dim1, out_dim2, out_dim3))
        return F.upsample(self.features(x_down), size=(inp_size[2], inp_size[3], inp_size[4]), mode='trilinear')


##############################################################################
# FCN3dNET
##############################################################################
class block(nn.Module):
    def __init__(self, conv_in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(block, self).__init__()

        self.layer = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm3d(out_channels),
                                    nn.ReLU(inplace=False))

    def forward(self, x):
        output = self.layer(x)
        return output


class stem(nn.Module):
    def __init__(self, conv_in_channels):
        super(stem, self).__init__()

        self.layer1 = block(conv_in_channels, 32, (3,3,1), 2)

        self.layer2 = block(32, 64, (3,3,4))

        self.path1 = nn.Sequential(
            block(64, 64, 1),
            block(64, 96, (3,3,1))
        )

        self.path2 = nn.Sequential(block(64,64,1),
                                   block(64,64,(1,7,1), 1, (1,2,0)),
                                   block(64,64,(7,1,1), 1, (2,1,0)),
                                   block(64,96,(3,3,1))
                                   )

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3_1 = self.path1(conv2)
        conv3_2 = self.path2(conv2)
        out = torch.cat((conv3_1, conv3_2), dim=1) #not sure
        return out


class inceptionA(nn.Module):
    def __init__(self, conv_in_channels):
        super(inceptionA, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = block(conv_in_channels, 192, (3,3,1), 2)

        self.layer2_2 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=2)

        self.layer3_1 = nn.Sequential(block(384, 32, 1),
                                      block(32, 384, 1)
                                      )


        self.layer3_2 = nn.Sequential(block(384, 32, 1),
                                      block(32, 32, (3, 3, 1), 1, (1, 1, 0)),
                                      block(32, 384, 1))

        self.layer3_3 = nn.Sequential(block(384, 32, 1),
                                      block(32, 48, (3,3,1), 1, (1,1,0)),
                                      block(48, 64, (3,3,1), 1, (1,1,0)),
                                      block(64, 384, 1)
                                      )

    def forward(self, x):
        # log('inceptionA - input: ' + str(x.size()))
        relu = self.layer1(x)
        # log('inceptionA - after layer1: ' + str(relu.size()))
        conv1 = self.layer2_1(relu)
        # log('inceptionA - after layer2_1: ' + str(conv1.size()))
        conv2 = self.layer2_2(relu)
        # log('inceptionA - after layer2_2: ' + str(conv2.size()))
        concat = torch.cat((conv1, conv2), dim=1)
        # log('inceptionA - after concat: ' + str(concat.size()))
        path3_1 = self.layer3_1(concat)
        # log('inceptionA - after layer3_1: ' + str(path3_1.size()))
        path3_2 = self.layer3_2(concat)
        # log('inceptionA - after layer3_2: ' + str(path3_2.size()))
        path3_3 = self.layer3_3(concat)
        # log('inceptionA - after layer3_3: ' + str(path3_3.size()))
        out = torch.add(path3_1, path3_2)
        # log('inceptionA - after concat first two: ' + str(out.size()))
        out = torch.add(out, path3_3)
        # log('inceptionA - after concat the 3rd one: ' + str(out.size()))
        out = torch.add(out, concat)
        # log('inceptionA - after concat the residual one: ' + str(out.size()))

        # residual- add input after relu to the final output

        return out

class reductionA(nn.Module):
    def __init__(self, conv_in_channels):
        super(reductionA, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=2)
        self.layer2_2 = block(conv_in_channels, 128, (3,3,1), 2)

        self.layer2_3 = nn.Sequential(
            block(conv_in_channels, 256, 1, 1),
            block(256, 256, (3, 3, 1), 1),
            block(256, 384, (3, 3, 1), 2, (1, 1, 0))
        )

    def forward(self, x):
        # log('reductionA - input: ' + str(x.size()))
        out = self.layer1(x)
        # log('reductionA - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        # log('reductionA - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        # log('reductionA - after layer2_2: ' + str(out2.size()))
        out3 = self.layer2_3(out)
        # log('reductionA - after layer2_3: ' + str(out3.size()))
        out = torch.cat((out1, out2), dim=1)
        # log('reductionA - after concate path1 and path2: ' + str(out.size()))
        out = torch.cat((out, out3), dim=1)
        # log('reductionA - after concate the 3rd path: ' + str(out.size()))
        return out

class inceptionB(nn.Module):
    def __init__(self, conv_in_channels):
        super(inceptionB, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)


        self.layer2_1 = nn.Sequential(
                                      block(conv_in_channels, 128, 1),
                                      block(128, 896, 1)
                                      )
        self.layer2_2 = nn.Sequential(
                                        block(conv_in_channels, 128, 1),
                                        block(128, 128, (1, 7, 1), 1, (1, 2, 0)),
                                        block(128, 128, (7, 1, 1), 1, (2, 1, 0)),
                                        block(128, 896, 1)
                                      )

    def forward(self, x):
        # log('inceptionB - input: ' + str(x.size()))
        out = self.layer1(x)
        # log('inceptionB - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        # log('inceptionB - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        # log('inceptionB - after layer2_2: ' + str(out2.size()))
        ret = torch.add(out1, out2)
        # log('inceptionB - after adding 2_1 & 2_2: ' + str(out.size()))
        out = torch.add(ret, out)
        # log('inceptionB - after adding residual path: ' + str(out.size()))

        # residual- add input after relu to the final output

        return out


class reductionB(nn.Module):
    def __init__(self, conv_in_channels):
        super(reductionB, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Sequential(block(conv_in_channels, 192, 1),
                                      block(192, 192, (3, 3, 1), 2)
                                      )
        self.layer2_2 = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=2)
        self.layer2_3 = nn.Sequential(block(conv_in_channels, 256, 1),
                                      block(256, 256, (1, 7, 1), 1, (1, 2, 0)),
                                      block(256, 320, (7,1,1), 1, (2, 1, 0)),
                                      block(320, 960, (3, 3, 1), 2)
                                      )

    def forward(self, x):
        # log('reductionB - input: ' + str(x.size()))
        out = self.layer1(x)
        # log('reductionB - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        # log('reductionB - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        # log('reductionB - after layer2_2: ' + str(out2.size()))
        out3 = self.layer2_3(out)
        # log('reductionB - after layer2_3: ' + str(out3.size()))
        out = torch.cat((out1, out2),dim=1)
        # log('reductionB - after concatenating layer2_1 & layer2_2: ' + str(out.size()))
        out = torch.cat((out, out3), dim=1)
        # log('reductionB - after concatenating  & layer2_3: ' + str(out.size()))

        return out


class inceptionC(nn.Module):
    def __init__(self, conv_in_channels):
        super(inceptionC, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Sequential(block(conv_in_channels, 192, 1),
                                      block(192, 2048, 1))
        self.layer2_2 = nn.Sequential(block(conv_in_channels, 192, 1),
                                      block(192, 224, (1, 3, 1), 1, (0, 1, 0)),
                                      block(224, 256, (3, 1, 1), 1, (1, 0, 0)),
                                      block(256, 2048, 1))

    def forward(self, x):
        # log('inceptionC - input: ' + str(x.size()))
        out = self.layer1(x)
        # log('inceptionC - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        # log('inceptionC - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        # log('inceptionC - after layer2_2: ' + str(out2.size()))

        # out = torch.cat((out1, out2),dim=1)
        # log('inceptionC - after concatenating layer2_1 & layer2_2: ' + str(out.size()))

        ret = torch.add(out1, out2)
        # log('inceptionC - after adding layer2_1 & layer2_2: ' + str(ret.size()))
        out = torch.add(ret, out) # residual- add input after relu to the final output
        # log('inceptionC - after adding residual path: ' + str(out.size()))

        return out