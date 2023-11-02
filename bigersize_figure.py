import argparse
import os

import torchvision.transforms as transforms
from torchvision.transforms import transforms
from DJSCC_DNA import *
from ssim import SSIM, MS_SSIM
import torch
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--rate_base', type=float, default=0.125, help='base per pixel')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--all_error', type=float, default=0.01, help='base error rate')
parser.add_argument('--sub_error', type=float, default=0.00215, help='base substitute error')
parser.add_argument('--del_error', type=float, default=0.002, help='base delete error')
parser.add_argument('--ins_error', type=float, default=0.00085, help='base insert error')
args = parser.parse_args()


# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

# Airplane  Butterfly

def readImage(path='Butterfly.png', size=256):  # 这里可以替换成自己的图片
    mode = Image.open(path)
    transform1 = transforms.Compose([
        # transforms.Resize(size),
        # transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode


def showTorchImage(image):
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    # plt.show()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

patch_size = 32
heigth = 256
width = 256

heigth2 = 288
width2 = 288

if __name__ == '__main__':
    # 图像分块到images_all，块大小是32*32
    mode = readImage()
    image = mode
    image = image.view(3, -1, patch_size, width).permute(1, 0, 2, 3)  # 横切，w不变
    images_all = image[0].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
    for i in range(1, int(heigth / patch_size)):
        images = image[i].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
        images_all = torch.cat([images_all, images], dim=0)

    # 把图像先扩大，然后分块到images_all2，块大小是32*32
    ZeroPad = nn.ZeroPad2d(padding=(16, 16, 16, 16))
    mode2 = ZeroPad(mode)
    image2 = mode2
    image2 = image2.view(3, -1, patch_size, width2).permute(1, 0, 2, 3)  # 横切，w不变
    images_all2 = image2[0].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
    for i in range(1, int(heigth2 / patch_size)):
        images2 = image2[i].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
        images_all2 = torch.cat([images_all2, images2], dim=0)

    # 定义模型结构
    (unit_num, c) = cul_para(args.rate_base)
    dna_encoder = Encoder_M(c=c)
    dna_decoder = Decoder_M(c=c)
    dna_channel = Channel_M()
    model = DNA_djscc_M(dna_encoder, dna_channel, dna_decoder)

    # 加载已训练的模型参数
    if args.resume:  # resume已训练过的模型path
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> model_best %s' % args.resume)
            model_best = torch.load(args.resume)
            start_epoch = model_best['epoch'] + 1
            best_test_loss = model_best['best_test_loss']
            model.load_state_dict(model_best['state_dict'])
            print('=> loaded model_best %s' % args.resume)
        else:
            print('=> no model_best at %s' % args.resume)
    else:
        print('=> no model_best at %s' % args.resume)

    # images_all编码解码过程
    loss_fn = nn.MSELoss(reduction='mean')
    ms_ssim = MS_SSIM()
    model.eval()

    b_x = images_all * 255.0
    with torch.no_grad():
        output_en, output_seg = model.encoder(b_x)
        output_channel = channel_test(output_seg, sub_error, del_error, ins_error)
        output_de = model.decoder(output_channel)
    decoded = torch.round(output_de)   # [0,255]

    # images_all2的编解码
    b_x2 = images_all2 * 255.0
    with torch.no_grad():
        output_en2, output_seg2 = model.encoder(b_x2)
        output_channel2 = channel_test(output_seg2, sub_error, del_error, ins_error)
        output_de2 = model.decoder(output_channel2)
    decoded2 = torch.round(output_de2)  # [0,255]


    # 图像1的解码图像拼接
    decoded = decoded/255   # [0,1]
    images_r = decoded.permute(1, 0, 2, 3)  # 交换维度
    images_r = images_r.reshape(-1, width, patch_size).permute(0, 2, 1)
    images_r = images_r.reshape(-1, heigth, width)
    # 图像2的解码图像拼接
    decoded2 = decoded2 / 255  # [0,1]
    images_r2 = decoded2.permute(1, 0, 2, 3)  # 交换维度
    images_r2 = images_r2.reshape(-1, width2, patch_size).permute(0, 2, 1)
    images_r2 = images_r2.reshape(-1, heigth2, width2)
    images_r2 = images_r2[:, 16:-16, 16:-16]

    # 图像融合
    images_r_last = torch.add(images_r, images_r2)/2

    # 计算前后psnr
    mse_loss = loss_fn(mode*255, images_r_last*255)
    psnr =  (10 * torch.log10_(255 * 255 / mse_loss))
    ssim_i = ms_ssim(mode.unsqueeze(0)*255, images_r_last.unsqueeze(0)*255)
    print("PSNR2", psnr)
    print("MS-SSIM2", ssim_i)

    toPIL = transforms.ToPILImage()
    pic = toPIL(images_r_last)
    pic.save('../simulation data./pictures./Butterfly_JSCC_rate%d-psnr%d-ssim%d.png'%(args.rate_base,psnr,ssim_i))

    # 画图
    plt.subplot(1,2,1)
    showTorchImage(mode)  # 原图
    plt.subplot(1,2,2)
    showTorchImage(images_r_last)  # 不重叠分块 psnr 22~23左右
    plt.show()
'''

    # 把图像先扩大，然后分块到images_all2，块大小是32*32
    ZeroPad = nn.ZeroPad2d(padding=(8, 24, 8, 24))
    mode3 = ZeroPad(mode)
    image3 = mode3
    image3 = image3.view(3, -1, patch_size, width2).permute(1, 0, 2, 3)  # 横切，w不变
    images_all3 = image3[0].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
    for i in range(1, int(heigth2 / patch_size)):
        images3 = image3[i].permute(0, 2, 1).view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)  # 竖切，w不变
        images_all3 = torch.cat([images_all3, images3], dim=0)

    # images_all3的编解码
    b_x3 = images_all3 * 255.0
    with torch.no_grad():
        output_en3, output_seg3 = model.encoder(b_x3)
        output_channel3 = channel_test(output_seg3, sub_error, del_error, ins_error)
        output_de3 = model.decoder(output_channel3)
    decoded3 = torch.round(output_de3)  # [0,255]

    # 图像2的解码图像拼接
    decoded3 = decoded3 / 255  # [0,1]
    images_r3 = decoded3.permute(1, 0, 2, 3)  # 交换维度
    images_r3 = images_r3.reshape(-1, width2, patch_size).permute(0, 2, 1)
    images_r3 = images_r3.reshape(-1, heigth2, width2)
    images_r3 = images_r3[:, 8:-24, 8:-24]
    
    # images_r_last = torch.add(images_r_last1, images_r3)/2
    
        #mse_loss = loss_fn(mode*255, images_r_last*255)
    #psnr =  (10 * torch.log10_(255 * 255 / mse_loss))
    #ssim_i = ms_ssim(mode.unsqueeze(0)*255, images_r_last.unsqueeze(0)*255)
    #print("PSNR2", psnr)
    #print("MS-SSIM2", ssim_i)
'''