import argparse
import os

class Options():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # model parameters
        parser.add_argument('--netG', type=str, default='resnet', help='only support resnet for now')
        parser.add_argument('--netD', type=str, default='PatchGAN', help='[GAN | PatchGAN | PixelGAN]')
        parser.add_argument('--upsample_factor', type=int, default=2, help='Up sampling factor')
        parser.add_argument('--ndf', default=32, type=int, help='number of filters in discriminator')
        parser.add_argument('--ngf', default=64, type=int, help='number of filters in generator')
        parser.add_argument('--generatorLR', type=float, default=0.001, help='learning rate for generator')
        parser.add_argument('--discriminatorLR', type=float, default=0.001, help='learning rate for discriminator')
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
        parser.add_argument('--gdl', type=float, default=1e-7, help='weight for gdl loss')
        parser.add_argument('--advW', type=float, default=0.1, help='weight of adv loss in total loss')

        # basic parameters
        parser.add_argument('--dataroot', type=str, default='./datasets')
        parser.add_argument('--output', type=str, default='./output')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--generatorWeights', type=str, default='./checkpoints/g.pth',
                            help="path to generator weights (to continue training)")
        parser.add_argument('--discriminatorWeights', type=str, default='./checkpoints/d.pth',
                            help="path to discriminator weights (to continue training)")
        parser.add_argument('--save_fre', type=int, default=10, help='checkpoint save frequency')

        # dataset parameters
        parser.add_argument('--file_extension', default='.hdr', help='image extension of your dataset')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch(patch) size = patch_size * num_gpus')
        parser.add_argument('--num_patches', default=8, type=int, help='only support 8 for now')
        parser.add_argument('--margin', default=4, type=int, help=' ')
        parser.add_argument('--img_width', default=192, type=int, help=' ')
        parser.add_argument('--img_height', default=144, type=int, help=' ')
        parser.add_argument('--img_depth', default=256, type=int, help=' ')
        parser.add_argument('--img_channel', default=1, type=int, help=' ')

        #training parameters
        parser.add_argument('--nEpochs', default=200, type=int, help='number of epochs to train for')
        parser.add_argument('--resume', default=0, type=int, help='resume training or not default:0/not')


        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        if opt.gpu_ids != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        return opt





