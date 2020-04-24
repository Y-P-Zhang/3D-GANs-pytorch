import os
import numpy as np
import glob
import nibabel as nib
from scipy.ndimage.interpolation import zoom

from options import Options

"""DownSample iSeg to get low-resolution image
    the DownSampled images will be stored in your hard disk at "dataroot-low"
   Use to visualize"""

NUMBER = 2 # choose number of images to DownSample
LABEL = 0 # LABEL=0: Do not downSample labels
         # LABEL != 0 : Only downSample labels

def get_img_list(dataroot,label=0,num=float("Inf")):
    """
    Parameters:
        label: choose to use label or T1/T2
        num :
    """
    img_list = []
    images = glob.glob(dataroot + "/*.hdr")
    if label:
        for img in images:
            if img.split('.')[-2][-5:] == 'label':
                img_list.append(img)
    else:
        for img in images:
            if img.split('.')[-2][-5:] != 'label':
                img_list.append(img)
    return img_list[:min(num, len(img_list))]


def read_data(f):
    data = nib.load(f)
    return np.squeeze(data.get_fdata())

def save_imgs(file_list,saveroot):
    for file in file_list:
        fdata = read_data(file)
        low_img = nib.Nifti1Image(zoom(fdata, 0.5, order=2), affine=None)
        imgname, _ = os.path.splitext(os.path.basename(file))
        img2save = saveroot+'/'+imgname+'.nii'
        nib.save(low_img, img2save)
        print(img2save, 'is saved')



if __name__ == '__main__':
    opt = Options().parse()
    dataroot = '.' + opt.dataroot
    saveroot = dataroot + '-low'
    mkdir(saveroot)
    file_list = get_img_list(dataroot,label=LABEL,num=NUMBER)
    save_imgs(file_list,saveroot)

