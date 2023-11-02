import random

import numpy
import numpy as np
import torchvision
from torch import zeros, ones
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import torch
import os
import shutil

batch_size = 64


def save_checkpoint(state, is_best, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'best_model.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


# - - - - - - - - - - data_50peoch reading  - - - - - - - - - -#
# open the list of cifar-10
def open_cifar10():
    train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=False)
    test_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def loss_fc2(x, b_loss, wd, gc_rate=0.50):
    loss_fn = nn.MSELoss(reduction='mean')
    E = torch.mean(x[:, 0:0 + wd], 1).unsqueeze(1)
    D = torch.var(x[:, 0:0 + wd], 1).unsqueeze(1)
    for j in range(int(wd / 2) - 1, x.shape[1] - int(wd / 2), int(wd / 2)):
        E_i = torch.mean(x[:, j:j + wd], 1).unsqueeze(1)
        D_i = torch.var(x[:, j:j + wd], 1).unsqueeze(1)
        E = torch.cat([E, E_i], dim=1)
        D = torch.cat([D, D_i], dim=1)
    holp_E = torch.ones(E.shape) * (1.5)
    holp_D = torch.ones(D.shape) * (2.25 - 2 * gc_rate)
    loss1 = loss_fn(E, holp_E)
    loss2 = loss_fn(D, holp_D)
    return loss1 + b_loss * loss2  # torch.mean(torch.add(temp, temp2))


def My_loss(encoder_output, decoder_output, target_output, a_loss, b_loss, wd):
    loss_fn = nn.MSELoss(reduction='mean')
    loss1 = loss_fn(decoder_output, target_output)
    loss2 = a_loss*loss_fc2(encoder_output, b_loss, wd)
    loss_all = loss1+loss2
    return loss_all, loss1


def cul_psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def succofx(x):
    diff = torch.abs(torch.add(x[:, :-1], torch.neg(x[:, 1:])))
    diff2 = torch.add(diff[:, :-1], (diff[:, 1:]))
    diff3 = torch.add(diff2[:, :-1], (diff2[:, 1:]))
    diff4 = torch.add(diff3[:, :-1], (diff3[:, 1:]))
    diff5 = torch.add(diff4[:, :-1], (diff4[:, 1:]))
    diff6 = torch.add(diff5[:, :-1], (diff5[:, 1:]))
    diff7 = torch.add(diff6[:, :-1], (diff6[:, 1:]))
    d1nm0 = ones(diff.shape).sum()-torch.where(zeros(diff.shape) < diff, ones(diff.shape), zeros(diff.shape)).sum()
    d2nm0 = ones(diff2.shape).sum()-torch.where(zeros(diff2.shape) < diff2, ones(diff2.shape), zeros(diff2.shape)).sum()
    d3nm0 = ones(diff3.shape).sum()-torch.where(zeros(diff3.shape) < diff3, ones(diff3.shape), zeros(diff3.shape)).sum()
    d4nm0 = ones(diff4.shape).sum()-torch.where(zeros(diff4.shape) < diff4, ones(diff4.shape), zeros(diff4.shape)).sum()
    d5nm0 = ones(diff5.shape).sum()-torch.where(zeros(diff5.shape) < diff5, ones(diff5.shape), zeros(diff5.shape)).sum()
    d6nm0 = ones(diff6.shape).sum()-torch.where(zeros(diff6.shape) < diff6, ones(diff6.shape), zeros(diff6.shape)).sum()
    d7nm0 = ones(diff7.shape).sum()-torch.where(zeros(diff7.shape) < diff7, ones(diff7.shape), zeros(diff7.shape)).sum()

    number = x.shape[0]
    l1 = x.shape[0] * x.shape[1] - d1nm0 - (d1nm0 - d2nm0)
    l2 = d1nm0 - 2 * d2nm0 + d3nm0
    l3 = d2nm0 - 2 * d3nm0 + d4nm0
    l4 = d3nm0 - 2 * d4nm0 + d5nm0
    l5 = d4nm0 - 2 * d5nm0 + d6nm0
    l6 = d5nm0 - 2 * d6nm0 + d7nm0
    l7 = d6nm0
    # x.shape[0]*x.shape[1]-d1nm0-(d1nm0-d2nm0): 均聚物1个的个数
    # d1nm0-2*d2nm0+d3nm0 : 均聚物2个的个数
    # d2nm0-2*d3nm0+d4nm0 : 均聚物3个的个数
    # d3nm0-2*d4nm0+d5nm0 : 均聚物4个的个数
    # d4nm0-2*d5nm0+d6nm0 : 均聚物5个的个数
    # d5nm0  ：均聚物6个及以上个的个数
    return l1/number, l2/number, l3/number, l4/number, l5/number, l6/number, l7/number



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True