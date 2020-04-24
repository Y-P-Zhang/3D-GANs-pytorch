import torch
from collections import OrderedDict



def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def cal_patched_shape(opt):
    batch_size = opt.batch_size
    patch_size = opt.num_patches
    img_depth = opt.img_depth
    img_height = opt.img_height
    img_width = opt.img_width
    margin = opt.margin
    patch_depth = img_depth//2 + margin
    patch_height = img_height//2 + margin
    patch_width = img_width//2 + margin
    channel = 1
    return [batch_size, patch_size, channel, patch_depth, patch_height, patch_width]
