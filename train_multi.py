import random
from matplotlib import pyplot as plt
import time
import argparse
from torchvision.utils import save_image
from functions import *
from DJSCC_DNA_multi import *
# - - - - - - -   initialize the parameters   - - - - - - - -#
# 需要注意的参数有：压缩率 信道参数 是否首次训练 模型存放路径
# 设置模型运行的设备
from ssim import SSIM, MS_SSIM

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# 设置默认参数
# C1:all_error = 0.005   0.00215+0.002+0.00085
# C2:all_error = 0.01    0.0043 +0.004+0.0017
# L1=a200b1   默认
# L2=a200b10
# (0.125, 0.25, 0.5, 0.75, 1, 1.5)
aerror = 0.005
suerror = aerror * 0.43
deerror = aerror * 0.4
inerror = aerror * 0.17
parser = argparse.ArgumentParser(description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--rate_base', type=float, default=0.125, help='base per pixel')
parser.add_argument('--all_error', type=float, default=aerror, help='base error rate')
parser.add_argument('--sub_error', type=float, default=suerror, help='base substitute error')
parser.add_argument('--del_error', type=float, default=deerror, help='base delete error')
parser.add_argument('--ins_error', type=float, default=inerror, help='base insert error')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train(default: 50)')
parser.add_argument('--test_every', type=int, default=5, metavar='N', help='test after every epochs(default: 5)')
parser.add_argument('--save_dir', type=str, default='./L75__C1__r0_125__multi', metavar='N', help='model saving directory')
parser.add_argument('--resume', type=str, default='./L75__C1__r0_125__multi/checkpoint.pth', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--result_dir', type=str, default='./Result__L75__C1__r0_5__multi', metavar='DIR', help='output directory')
parser.add_argument('--a_loss', type=int, default=75, help='loss function parameter')
parser.add_argument('--b_loss', type=int, default=10, help='loss function parameter')
parser.add_argument('--wd', type=int, default=8, help='loss function parameter')
parser.add_argument('--seed', type=int, default=2023, help='learning rate(default: 0.001)')
args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def test(model, optimizer, test_dataloader, epoch, best_test_loss):
    model.eval()
    test_avg_loss = 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_dataloader):
            b_x = x.permute((0, 1, 2, 3)) * 255.0
            b_y = x.permute((0, 1, 2, 3)) * 255.0

            (unit_num, c)=cul_para(args.rate_base)
            output_en, output_seg = model.encoder(b_x, unit_num=unit_num, seg_length=256)
            output_channel = channel_test(output_seg, args.sub_error, args.del_error, args.ins_error)
            output_de = model.decoder(output_channel, c)
            test_loss, MSE = My_loss(output_en, output_de, b_y, args.a_loss, args.b_loss, args.wd)
            test_avg_loss += test_loss

            if step % (len(test_dataloader) // 5) == 0:
                print("epoch={},{}/{}of valid, mse loss={}".format(epoch, step, len(test_dataloader), MSE.item()))

        test_avg_loss /= len(test_dataloader.dataset)

        '''保存目前训练好的模型'''
        # 保存模型
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,  # 迭代次数
            'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
            'state_dict': model.state_dict(),  # 当前训练过的模型的参数
            'optimizer': optimizer.state_dict(), # 当前优化器的参数
        }, is_best, args.save_dir)

        return best_test_loss


def main():
    train_loss = []
    train_epochs_loss = []
    # 加载数据集
    train_dataloader, test_dataloader = open_cifar10()
    # 创建网络模型
    (unit_num, c) = cul_para(args.rate_base)
    dna_encoder = Encoder_M(c=c)
    dna_decoder = Decoder_M(c=c)
    dna_channel = Channel_M()
    model = DNA_djscc_M(dna_encoder, dna_channel, dna_decoder)
    # summary(model, input_size=[(3, 32, 32)])
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # 查看是否存在已经训练过，检查点
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:   # resume已训练过的模型path
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    # 查看文件是否存在
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # train
    # setup_seed(args.seed)
    for epoch in range(start_epoch, args.epochs):
        since = time.time()
        # 训练开始
        model.train()
        train_epoch_loss = []
        print("-------第{}轮训练开始-------".format(epoch + 1))
        #for p in optimizer.param_groups:
        #    if p['lr'] > 0.00001:
        #        p['lr'] = 0.01 / (2 ** (math.floor(epoch / 5)))  # 根据epoch次数改变lr
        for p in optimizer.param_groups:
            if epoch >= 40:
                p['lr'] = 0.0005
            elif epoch >= 30:
                p['lr'] = 0.00075
            elif epoch >= 20:
                p['lr'] = 0.001
            elif epoch >= 10:
                p['lr'] = 0.0025
            elif epoch >= -1:
                p['lr'] = 0.005


        for step, (x, y) in enumerate(train_dataloader):
            b_x = x.permute((0, 1, 2, 3)) * 255.0
            b_y = x.permute((0, 1, 2, 3)) * 255.0

            encoder_output, channel_seg, decoder_output = model(b_x, args.sub_error, args.del_error, args.ins_error, args.rate_base)
            loss, MSE = My_loss(encoder_output, decoder_output, b_y, args.a_loss, args.b_loss, args.wd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % (len(train_dataloader) // 2) == 0:
                time_elapsed = time.time() - since
                print('Epoch [{}/{}] : Total-loss = {:.4f}, MSE-Loss = {:.4f}, BLC-loss = {:.4f}, '
                      'lr= {:.6f}, r={:.4f}, er={:.4f}'
                      .format(epoch + 1, args.epochs, loss.data, MSE.data, loss - MSE,
                              optimizer.state_dict()['param_groups'][0]['lr'], args.rate_base, args.all_error),
                      'Training complete in {:.0f}m {:.0f}s'.format(
                          time_elapsed // 60, time_elapsed % 60)
                      )
                ll1, ll2, ll3, ll4, ll5, ll6, ll7 = succofx(channel_seg)
                (base, number) = torch.unique(channel_seg, return_counts=True)  # 计算AGCT数量
                AGCT_rate = div(number, sum(number))
                ssim = SSIM()
                ms_ssim = MS_SSIM()
                ssim_iando = ssim(decoder_output, b_y)
                msssim_iando = ssim(decoder_output, b_y)
                print("RLL567+:", ll5, ll6, ll7, "AG:", AGCT_rate,
                      "PSNR={:.4f}".format(20 * math.log10(255 / math.sqrt(MSE))), "ssim:{:.4f}".format(ssim_iando),
                      "ms-ssim:{:.4f}".format(msssim_iando))

            train_epoch_loss.append(MSE.item())
            train_loss.append(MSE.item())
            #if step == 0:
            #    # visualize reconstructed result at the beginning of each epoch
            #    x_concat = torch.cat([x.view(-1, 3, 32, 32), (decoder_output/255).view(-1, 3, 32, 32)], dim=3)
            #    save_image(x_concat, './%s/reconstructed-%d.png' % (args.result_dir, epoch + 1))

        train_epochs_loss.append(np.average(train_epoch_loss))

        # 测试模型并保存检查点
        if (epoch + 1) % args.test_every == 0:
            best_test_loss = test(model, optimizer, test_dataloader, epoch, best_test_loss)

    return train_epochs_loss


if __name__ == '__main__':
    start = time.perf_counter()
    loss_epoch = main()
    end = time.perf_counter()
    runTime = end - start
    print("train_loss", loss_epoch)
    print("训练及验证时长：", runTime)
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()



