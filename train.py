import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import interpolate

from models.build_model import build_netG, build_netD
from data.customdataset import CustomDataset
from models.losses import gdloss
from utils.util import new_state_dict
from options import Options

opt = Options().parse()
opt.phase = 'train'
opt.dataset = 'iseg'
print(opt)

data_set = CustomDataset(opt)
print('Image numbers:', data_set.img_size)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

generator = build_netG(opt)
discriminator, target_real, target_fake = build_netD(opt)

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)

if opt.resume:
    generator.load_state_dict(new_state_dict(opt.generatorWeights))
    discriminator.load_state_dict(new_state_dict(opt.discriminatorWeights))
    print('Weight is loaded')
else:
    pretrainW = './checkpoints/g_pre-train.pth'
    if os.path.exists(pretrainW):
        generator.load_state_dict(new_state_dict(pretrainW))
        print('Pre-Trained G Weight is loaded')

adversarial_criterion = nn.MSELoss()  # nn.BCELoss()

if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    generator.cuda()
    discriminator.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, weight_decay=1e-4)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, weight_decay=1e-4)
StepLR_G = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=10, gamma=0.85)
StepLR_D = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=10, gamma=0.85)

print('start training')
for epoch in range(opt.nEpochs):
    mean_generator_adversarial_loss = 0.0
    mean_generator_l2_loss = 0.0
    mean_generator_gdl_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):
        # get input data
        high_real_patches = data['high_img_patches']  # [batch_size,num_patches,C,D,H,W]
        for k in range(0, opt.num_patches):
            high_real_patch = high_real_patches[:, k]  # [BCDHW]
            low_patch = interpolate(high_real_patch, scale_factor=0.5)
            if use_gpu:
                high_real_patch = high_real_patch.cuda()
                # generate fake data
                high_gen = generator(low_patch.cuda())
            else:
                high_gen = generator(low_patch)

            ######### Train discriminator #########
            discriminator.zero_grad()

            discriminator_loss = 0.5 * adversarial_criterion(discriminator(high_real_patch), target_real) + \
                                 0.5 * adversarial_criterion(discriminator(high_gen.detach()), target_fake)

            mean_discriminator_loss += discriminator_loss
            discriminator_loss.backward()
            optim_discriminator.step()

            ######### Train generator #########
            generator.zero_grad()

            generator_gdl_loss = opt.gdl * gdloss(high_real_patch, high_gen)
            mean_generator_gdl_loss += generator_gdl_loss

            generator_l2_loss = nn.MSELoss()(high_real_patch, high_gen)
            mean_generator_l2_loss += generator_l2_loss

            generator_adversarial_loss = adversarial_criterion(discriminator(high_gen), target_real)
            mean_generator_adversarial_loss += generator_adversarial_loss

            generator_total_loss = generator_gdl_loss + generator_l2_loss + opt.advW * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss

            generator_total_loss.backward()
            optim_generator.step()

        ######### Status and display #########
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (GDL/L2/Adv/Total): %.4f/%.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, len(dataloader),
                discriminator_loss, generator_gdl_loss, generator_l2_loss,
                generator_adversarial_loss, generator_total_loss))

    StepLR_G.step()
    StepLR_D.step()

    if epoch % opt.save_fre == 0:
        # Do checkpointing
        torch.save(generator.state_dict(), '%s/g.pth' % opt.checkpoints_dir)
        torch.save(discriminator.state_dict(), '%s/d.pth' % opt.checkpoints_dir)

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (GDL/L2/Adv/Total): %.4f/%.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, len(dataloader),
            mean_discriminator_loss / len(dataloader) / opt.num_patches,
            mean_generator_gdl_loss / len(dataloader) / opt.num_patches,
            mean_generator_l2_loss / len(dataloader) / opt.num_patches,
            mean_generator_adversarial_loss / len(dataloader) / opt.num_patches,
            mean_generator_total_loss / len(dataloader) / opt.num_patches))
