import torch
import torch.nn as nn
import numpy as np

lrelu = nn.LeakyReLU(0.2)

def conv_block(ndf,stage):
    module = []
    module += [
        nn.Conv3d(ndf*(2**stage), ndf*(2**stage), kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(ndf*(2**stage)),
        nn.LeakyReLU(negative_slope=0.3),
        nn.Conv3d(ndf*(2**stage), ndf*(2**(stage+1)), kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(ndf*(2**(stage+1))),
        nn.LeakyReLU(negative_slope=0.3)
    ]
    return module


class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.input_channel = opt.img_channel
        self.ndf = opt.ndf  # number of filters
        self.batch_size = opt.batch_size
        # Discriminator input shape
        self.img_width = opt.img_width/2 + opt.margin
        self.img_height = opt.img_height/2 + opt.margin
        self.img_depth = opt.img_depth/2 + opt.margin
        self.outshape = torch.Size([self.batch_size,1])

        sequence = [nn.Conv3d(1, self.ndf, kernel_size=3, padding=1), lrelu]
        for i in range(3):
            sequence += conv_block(self.ndf, i)
        # use conv instead of fc layer
        sequence += [nn.Conv3d(self.ndf*8, self.ndf*8, kernel_size=3, stride=2, padding=1),
                     nn.BatchNorm3d(self.ndf*8),lrelu]
        sequence += [nn.Conv3d(self.ndf*8, 1, kernel_size=3, stride=2, padding=1)]
        shape = self.compute_conv_shape()
        sequence += [nn.MaxPool3d(kernel_size=shape[-3:])]
        sequence += [nn.Flatten()]
        self.model = nn.Sequential(*sequence)

    def compute_conv_shape(self):
        """compute tensor shape after the last conv
            [B,C,D,H,W]"""
        return [int(self.batch_size),
                int(self.input_channel*self.ndf*8),
                int(np.ceil(np.ceil(np.ceil(np.ceil(np.ceil(self.img_depth/2)/2)/2)/2)/2)),
                int(np.ceil(np.ceil(np.ceil(np.ceil(np.ceil(self.img_height/2)/2)/2)/2)/2)),
                int(np.ceil(np.ceil(np.ceil(np.ceil(np.ceil(self.img_width/2)/2)/2)/2)/2))]

    def forward(self,input):
        result = self.model(input)
        logits = nn.Sigmoid()(result)
        return logits



class PatchDiscriminator(nn.Module):
    def __init__(self,opt):
        super(PatchDiscriminator,self).__init__()
        self.input_channel = opt.img_channel
        self.ndf = opt.ndf  # number of filters
        self.batch_size = opt.batch_size
        # Discriminator input shape
        self.img_width = opt.img_width//2 + opt.margin
        self.img_height = opt.img_height//2 + opt.margin
        self.img_depth = opt.img_depth//2 + opt.margin
        self.outshape = self.compute_conv_shape()

        sequence = [nn.Conv3d(1, self.ndf, kernel_size=3, padding=1), lrelu]
        for i in range(3):
            sequence += conv_block(self.ndf, i)
        sequence += [nn.Conv3d(self.ndf*8, 1, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def compute_conv_shape(self):
        """compute tensor shape after the last conv
            [N,C,D,H,W]"""
        return [int(self.batch_size),
                self.input_channel,
                int(np.ceil(np.ceil(np.ceil(self.img_depth/2)/2)/2)),
                int(np.ceil(np.ceil(np.ceil(self.img_height/2)/2)/2)),
                int(np.ceil(np.ceil(np.ceil(self.img_width/2)/2)/2))]

    def forward(self, x):
        result = self.model(x)
        return result


class PixelDiscriminator(nn.Module):
    def __init__(self,opt):
        super(PixelDiscriminator,self).__init__()
        self.input_channel = opt.img_channel
        self.ndf = opt.ndf  # number of filters
        self.batch_size = opt.batch_size
        # Discriminator input shape
        self.img_width = opt.img_width//2 + opt.margin
        self.img_height = opt.img_height//2 + opt.margin
        self.img_depth = opt.img_depth//2 + opt.margin
        self.outshape = torch.Size([self.batch_size, self.input_channel, self.img_depth,
                                    self.img_height, self.img_width])

        sequence = [nn.Conv3d(1, self.ndf, kernel_size=1,), lrelu]
        sequence += [nn.Conv3d(self.ndf, 2*self.ndf, kernel_size=1), nn.BatchNorm3d(2*self.ndf), lrelu]
        sequence += [nn.Conv3d(2*self.ndf, 4*self.ndf, kernel_size=1), nn.BatchNorm3d(4*self.ndf), lrelu]
        sequence += [nn.Conv3d(4 * self.ndf, self.input_channel, kernel_size=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        result = self.model(x)
        return result