from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image
import argparse
from DJSCC_DNA import *
from functions import *

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

def readImage(path=''):  # 这里可以替换成自己的图片
    mode = Image.open(path)
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode


def showTorchImage(image):
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    plt.show()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

patch_size = 32
heigth = 32
width = 240

if __name__ == '__main__':
    # 生成原始图片并保存
    train_dataloader, test_dataloader = open_cifar10()
    # for batch_index, (x, _) in enumerate(test_dataloader):
    #    x = x[1,:,:,:]
    #    if batch_index < 8:
    #        save_image(x, '%s/cifer10-original-%d.png' % ('../simulation data./pictures', batch_index))
    loss_fn = nn.MSELoss(reduction='mean')
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

    # 将原始小图片进行拼接到images(0~1), inputi(0-255)
    images = torch.tensor([])
    for index in range(8):
        path = '%s/cifer10-original-%d.png' % ('../simulation data./pictures', index)
        image = readImage(path).unsqueeze(0)
        images = torch.cat([images, image], dim=0)
    inputi = images*255.0
    # 对inputi进行操作
    model.eval()
    with torch.no_grad():
        output_en, output_seg = model.encoder(inputi)
        output_channel = channel_test(output_seg, sub_error, del_error, ins_error)
        output_de = model.decoder(output_channel)
        mse_loss = loss_fn(output_de, inputi)
    psnr = 10 * torch.log10_(255 * 255 / mse_loss)
    print(psnr)

    # 分别保存操作前后的小图片们
    # save_image(images, '../simulation data./pictures/original.png')
    save_image(output_de/255.0, '../simulation data./pictures/djscc-rate%d-psnr%d.png'%(args.rate_base,psnr))

