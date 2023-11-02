import argparse
from functions import *
from DJSCC_DNA_multi import *


# - - - - - - -   initialize the parameters   - - - - - - - -#
# 需要注意的参数有：压缩率 信道参数 是否首次训练 模型存放路径
# 设置模型运行的设备
from ssim import SSIM

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# 设置默认参数
# C1:error_rate = 0.005
# C1_5:error_rate = 0.0075
aerror = 0.01
suerror = aerror * 0.43
deerror = aerror * 0.4
inerror = aerror * 0.17
parser = argparse.ArgumentParser(description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--rate_base', type=float, default=0.125, help='base per pixel')
parser.add_argument('--all_error', type=float, default=aerror, help='base error rate')
parser.add_argument('--sub_error', type=float, default=suerror, help='base substitute error')
parser.add_argument('--del_error', type=float, default=deerror, help='base delete error')
parser.add_argument('--ins_error', type=float, default=inerror, help='base insert error')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train(default: 100)')
parser.add_argument('--resume', type=str, default='./model_data_2/L175__C1__r0_125__multi/best_model.pth', metavar='PATH', help='path to latest checkpoint(default: None)')
# :resume: 已训练过的模型参数的path，eg: ./C1_r0125/checkPoint.pth
parser.add_argument('--seed', type=int, default=2023, help='learning rate(default: 0.001)')

args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def main():
    model.eval()
    loss_fn = nn.MSELoss(reduction='mean')
    train_dataloader, test_dataloader = open_cifar10()
    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0
    l5 = 0
    l6 = 0
    l7 = 0
    AGCT_rate_avg = 0
    psnr_avg = 0
    ssim_avg = 0
    for step, (x, y) in enumerate(test_dataloader):
        setup_seed(args.seed)
        b_x = x.permute((0, 1, 2, 3)) * 255.0
        b_y = x.permute((0, 1, 2, 3)) * 255.0
        with torch.no_grad():
            (unit_num, c) = cul_para(args.rate_base)
            output_en, output_seg = model.encoder(b_x, unit_num=unit_num, seg_length=256)
            output_channel = channel_test(output_seg, args.sub_error, args.del_error, args.ins_error)
            output_de = model.decoder(output_channel, c)
            mse_loss = loss_fn(output_de, b_y)

        # 转换AGCT
        #want = output_seg.numpy()
        #d = {0: 'a', 1: 'g', 2: 'c', 3: 't'}
        #a2 = np.vectorize(d.get)(want)

         # 统计序列特征
        ll1, ll2, ll3, ll4, ll5, ll6, ll7 = succofx(output_seg)
        (base, number) = torch.unique(output_seg, return_counts=True)  # 计算AGCT数量
        AGCT_rate = div(number, sum(number))
        psnr = 10 * torch.log10_(255 * 255 / mse_loss)
        ssim = SSIM()
        ssim_io = ssim(output_de, b_y)
        print('step:', step, "test-psnr:", psnr)
        l1 += ll1
        l2 += ll2
        l3 += ll3
        l4 += ll4
        l5 += ll5
        l6 += ll6
        l7 += ll7
        AGCT_rate_avg += AGCT_rate
        psnr_avg += psnr
        ssim_avg += ssim_io
    AGCT_rate_avg /= (step+1)
    l1 /= (step+1)
    l2 /= (step+1)
    l3 /= (step+1)
    l4 /= (step+1)
    l5 /= (step+1)
    l6 /= (step+1)
    l7 /= (step+1)
    psnr_avg /= (step+1)
    ssim_avg /= (step + 1)
    print("rate base_ = ", args.rate_base)
    print("error rate = ", args.all_error)
    print('PSNR =', psnr_avg.tolist())
    print('SSIM =', ssim_avg.tolist())
    print('homologous =', "[", l1.tolist(), ",", l2.tolist(), ",", l3.tolist(), ",", l4.tolist(), ",", l5.tolist(), ",", l6.tolist(), ",", l7.tolist(), "]")  # 输出连续情况
    print("AGCT_rate =", AGCT_rate.tolist())
    print("best epoch:", start_epoch)
    print('finish')



if __name__ == '__main__':
    # 创建网络模型
    (unit_num, c) = cul_para(args.rate_base)
    dna_encoder = Encoder_M(c=c)
    dna_decoder = Decoder_M(c=c)
    dna_channel = Channel_M()
    model = DNA_djscc_M(dna_encoder, dna_channel, dna_decoder)
    if args.resume:  # resume已训练过的模型path
        if os.path.isfile(args.resume):
            # 载入已经训练过的模型参数与结果
            print('=> model_best %s' % args.resume)
            model_best = torch.load(args.resume)
            start_epoch = model_best['epoch'] + 1
            best_test_loss = model_best['best_test_loss']
            model.load_state_dict(model_best['state_dict'])
            print('=> loaded model_best %s' % args.resume)
            main()
        else:
            print('=> no model_best at %s' % args.resume)
    else:
        print('=> no path at %s' % args.resume)
