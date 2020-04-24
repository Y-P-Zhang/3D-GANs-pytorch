import torch

from .discriminators import Discriminator,PatchDiscriminator,PixelDiscriminator
from .generators import ResnetGenerator


def build_netG(opt):
    if opt.netG == 'resnet':
        generator = ResnetGenerator(upsample_factor=opt.upsample_factor, n_residual_blocks=6, deup=True)
    else:
        raise NotImplementedError
    return generator

def build_netD(opt):
    if opt.netD == 'GAN':
        discriminator = Discriminator(opt)
    elif opt.netD == 'PatchGAN':
        discriminator = PatchDiscriminator(opt)
    elif opt.netD == 'PixelGAN':
        discriminator = PixelDiscriminator(opt)
    else:
        raise NotImplementedError

    target_real = torch.ones(discriminator.outshape)
    target_fake = torch.zeros(discriminator.outshape)
    return discriminator, target_real, target_fake