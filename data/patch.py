import numpy as np
from skimage.util import view_as_windows


def make_patches(img, margin=16, num_patches=8):
    """make patches for single image
    img : shape of [HWD1]
    """
    img = np.squeeze(img)
    h, w, d = img.shape
    assert num_patches in [8, 16, 32, 64], "num_patches must in [8,16,32,64]"

    if num_patches == 8:
        h, w, d = h // 2, w // 2, d // 2
    if num_patches == 16:
        h, w, d = h // 2, w // 2, d // 4
    if num_patches == 32:
        h, w, d = h // 2, w // 4, d // 4
    if num_patches == 64:
        h, w, d = h // 4, w // 4, d // 4

    patches = np.empty(
        [num_patches,
         1,
         d + margin,
         h + margin,
         w + margin], dtype=np.float32)
    patch = view_as_windows(np.ascontiguousarray(img), window_shape=(h + margin, w + margin, d + margin),
                            step=(h - margin, w - margin, d - margin))

    i = 0
    for d in range(patch.shape[0]):
        for v in range(patch.shape[1]):
            for h in range(patch.shape[2]):
                p = patch[d, v, h, :]
                p = p.transpose((2, 0, 1))  # DHW
                p = p[np.newaxis]  # CDHW
                patches[i] = p
                i = i + 1
    return patches


def depatched(patches, opt, margin=16):
    """depatch patches for a batch of image
       patches :numpy array, shape of [batch_size,num_patches,C,D,H,W]
    """
    patches = patches.transpose((0, 1, 4, 5, 3, 2))  # to [B,n_p,H,W,D,C]

    depth = int(opt.img_depth//2)
    height = int(opt.img_height//2)
    width = int(opt.img_width//2)
    assert (patches.shape[1] == 8), "This function only support num_patches == 8"

    img = np.empty([patches.shape[0], height * 2, width * 2, depth * 2, 1],
                   dtype=np.int)  # [batch_size,num_patches,H,W,D,C]
    img[:, :height, :width, :depth, :] = patches[:, 0, :height, :width, :depth, :]
    img[:, :height, :width, depth:, :] = patches[:, 1, :height, :width, margin:, :]
    img[:, :height, width:, :depth, :] = patches[:, 2, :height, margin:, :depth, :]
    img[:, :height, width:, depth:, :] = patches[:, 3, :height, margin:, margin:, :]
    img[:, height:, :width, :depth, :] = patches[:, 4, margin:, :width, :depth, :]
    img[:, height:, :width, depth:, :] = patches[:, 5, margin:, :width, margin:, :]
    img[:, height:, width:, :depth, :] = patches[:, 6, margin:, margin:, :depth, :]
    img[:, height:, width:, depth:, :] = patches[:, 7, margin:, margin:, margin:, :]

    return img
