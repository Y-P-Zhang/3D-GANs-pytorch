import torch.utils.data as data
import nibabel as nib
import numpy as np
import glob

from data.patch import make_patches


def make_datalist(opt):
    if opt.phase == 'train':
        img_list_high = glob.glob(opt.dataroot + "/train/*" + opt.file_extension)
    elif opt.phase == 'test':
        img_list_high = glob.glob(opt.dataroot + "/test/*" + opt.file_extension)
    return sorted(img_list_high)

def transform(x):
    # only for iseg dataset
    y = (x/1000 - np.mean(x/1000)) / np.std(x/1000)
    return y

class CustomDataset(data.Dataset):
    def __init__(self, opt):
        self.img_paths = make_datalist(opt)
        self.img_size = len(self.img_paths)
        self.opt = opt

    def __getitem__(self, index):
        img_path = self.img_paths[index % self.img_size]
        high_img = np.array(nib.load(img_path).get_fdata()).astype(np.float32)[:, 18:178, 75:203, :]  # [H,W,D,1]
        high_img_patches = make_patches(high_img, margin=self.opt.margin, num_patches=self.opt.num_patches)
        high_img_patches = transform(high_img_patches)

        return {'high_img_patches': high_img_patches, 'img_path': img_path}

    def __len__(self):
        return self.img_size


