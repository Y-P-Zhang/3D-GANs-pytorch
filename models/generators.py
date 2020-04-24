import torch.nn as nn

lrelu = nn.LeakyReLU(0.2)

class ResnetBlock(nn.Module):
    def __init__(self, inf, onf):
        """
        Parameters:
            inf: input number of filters
            onf: output number of filters
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(inf, onf)

    def build_conv_block(self, inf, onf):
        conv_block = [nn.Conv3d(inf, onf, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(onf), lrelu]
        conv_block += [nn.Conv3d(inf, onf, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(onf)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DeUpBlock(nn.Module):
    """Up sample block using torch.nn.ConvTranspose3d"""
    def __init__(self, inf, onf):
        super(DeUpBlock, self).__init__()
        sequence = [nn.ConvTranspose3d(inf, onf, kernel_size=6, stride=2, padding=2), lrelu]
        self.deupblock = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.deupblock(x)
        return out



class UpBlock(nn.Module):
    """Up sample block using torch.nn.Upsample
    """
    def __init__(self, inf, onf):
        super(UpBlock, self).__init__()
        sequence = [nn.Conv3d(inf, onf, kernel_size=3, padding=1), lrelu]
        sequence += [nn.Upsample(scale_factor=2)]
        sequence += [nn.Conv3d(inf, onf, kernel_size=3, padding=1), lrelu]
        self.upblock = nn.Sequential(*sequence)

    def forward(self, x):
        return self.upblock(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=32, n_residual_blocks=6, upsample_factor=2, deup=False):
        """
        Parameters:
            n_blocks: the number of resnetblocks
            deup: use deconv to upsample
        """
        assert upsample_factor % 2 == 0, "only support even upsample_factor"
        super(ResnetGenerator,self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.ngf = ngf
        self.deup = deup
        # the first conv-lrelu
        self.conv_blockl_1 = nn.Sequential(nn.Conv3d(input_nc,ngf,kernel_size=3,padding=1),lrelu)
        # residual blocks
        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), ResnetBlock(ngf,ngf))
        # the conv after residual blocks
        self.conv_blockl_2 = nn.Sequential(nn.Conv3d(ngf, ngf, kernel_size=3, padding=1),
                                           nn.BatchNorm3d(ngf))
        # upsample blocks
        for i in range(int(self.upsample_factor/2)):
            if self.deup:
                self.add_module('de_upsample' + str(i+1), DeUpBlock(ngf, ngf))
            else:
                self.add_module('upsample' + str(i+1), UpBlock(ngf, ngf))
        # the last conv
        self.conv3 = nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_blockl_1(x)
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)
        # large skip connection
        x = self.conv_blockl_2(y) + x

        for i in range(int(self.upsample_factor/2)):
            if self.deup:
                x = self.__getattr__('de_upsample' + str(i + 1))(x)
            else:
                x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)


# TODO 3DUnet

# TODO ResUnet







