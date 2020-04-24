import torch
"""
def gdloss(real,fake):
    dx_real = real[:, :, :, 1:, :] - real[:, :, :, :-1, :] #[BCHWD]
    dy_real = real[:, :, 1:, :, :] - real[:, :, :-1, :, :]
    dz_real = real[:, :, :, :, 1:] - real[:, :, :, :, :-1]
    dx_fake = fake[:, :, :, 1:, :] - fake[:, :, :, :-1, :]
    dy_fake = fake[:, :, 1:, :, :] - fake[:, :, :-1, :, :]
    dz_fake = fake[:, :, :, :, 1:] - fake[:, :, :, :, :-1]

    gd_loss = torch.sum(torch.pow(torch.abs(dx_real) - torch.abs(dx_fake),2),dim=(2,3,4)) + \
              torch.sum(torch.pow(torch.abs(dy_real) - torch.abs(dy_fake),2),dim=(2,3,4)) + \
              torch.sum(torch.pow(torch.abs(dz_real) - torch.abs(dz_fake),2),dim=(2,3,4))
    return torch.sum(gd_loss)"""

def gdloss(real,fake):
    dreal = real[:, :, 1:, 1:, 1:] - real[:, :, :-1, :-1, :-1] #[BCHWD]
    dfake = fake[:, :, 1:, 1:, 1:] - fake[:, :, :-1, :-1, :-1]
    gd_loss = torch.sum((torch.abs(dreal) - torch.abs(dfake))**2, dim=(0, 1, 2, 3, 4))

    return gd_loss

