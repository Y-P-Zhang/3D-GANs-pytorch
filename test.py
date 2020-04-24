#!/usr/bin/env python

import os
import nibabel as nib

import numpy as np
import torch
from torch.nn.functional import interpolate

from models.build_model import build_netG
from data.customdataset import CustomDataset
from data.patch import depatched
from utils.util import new_state_dict, cal_patched_shape

from options import Options
opt = Options().parse()
print(opt)

# TODO: Evaluation
opt.batch_size = 1
opt.phase = 'test'

try:
    os.makedirs(opt.output)
except OSError:
    pass


data_set = CustomDataset(opt)
print('Image numbers:', data_set.img_size)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))
generator = build_netG(opt)

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)

assert opt.discriminatorWeights != ''
assert opt.generatorWeights != ''

generator.load_state_dict(new_state_dict(opt.generatorWeights))

high_fake = np.empty(cal_patched_shape(opt))


if num_gpus > 0:
    generator.cuda()

print('Test started...')

# Set evaluation mode (not training)
generator.eval()

for i, data in enumerate(dataloader):
    # Generate data
    high_real_patches = data['high_img_patches']

    # Downsample images to low resolution
    for k in range(0, opt.num_patches):
        high_real_patch = high_real_patches[:, k]  # [BCDHW]
        low_patch = interpolate(high_real_patch, scale_factor=0.5)

        # generate fake high sr image
        if num_gpus >= 1:
            high_real_patch = high_real_patch.cuda()
            high_fake_patch = generator(low_patch.cuda())
        else:
            high_real_patch = high_real_patch
            high_fake_patch = generator(low_patch)
        high_fake[:, k] = high_fake_patch.cpu().detach().numpy()

    # save generated images
    high_fake = depatched(high_fake, opt)
    high_fake = np.squeeze(high_fake, axis=0)
    high_fake = nib.Nifti1Image(high_fake, affine=None)
    img_path = data['img_path']
    imgname, _ = os.path.splitext(os.path.basename(img_path[0]))
    img2save = opt.output + '/' + imgname+'_gen' + '.nii'
    nib.save(high_fake, img2save)
    print(img2save, 'is saved')
    high_fake = np.empty(cal_patched_shape(opt))

