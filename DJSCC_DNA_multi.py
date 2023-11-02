import torch
import torch.nn as nn
from torch import reshape, add, randint, rand, mul, stack, remainder, div, transpose, cat, argmax, flatten, softmax
from torch.nn import Sequential, Conv2d, PReLU, BatchNorm2d, Sigmoid, ConvTranspose2d, Conv1d, ConvTranspose1d

# rate_base: base per pixel
# pixel_num=1024 : the number of pixels
# seg_length=256 : the length of each oligo
# unit_num = int(pixel_num*rate_base) : the number of unit of conv output
# c = int(unit_num / 64)  : the last channel number of conv


class DNA_djscc_M(nn.Module):
    def __init__(self, encoder, channel, decoder):
        super(DNA_djscc_M, self).__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder

    def forward(self, image, sub_error, del_error, ins_error, rate_base, pixel_num=1024, seg_length=256):
        unit_num = int(pixel_num*rate_base)
        c = int(unit_num / 64)
        en_output, segment_en = self.encoder(image, unit_num, seg_length)     # [b, k] {0，1，2，3}
        channel_x_all = self.channel(segment_en, sub_error, del_error, ins_error)
        de_output = self.decoder(channel_x_all, c)
        return en_output, segment_en, de_output


class Encoder_M(nn.Module):
    def __init__(self, c):
        super(Encoder_M, self).__init__()

        self.convs = Sequential(
            Conv2d(3, 16, 3, stride=2, padding=1),
            PReLU(),
            Conv2d(16, 32, 3, stride=2, padding=1),
            BatchNorm2d(32),
            PReLU(),
            Conv2d(32, 32, 3, stride=1, padding=1),
            BatchNorm2d(32),
            PReLU(),
            Conv2d(32, 32, 3, stride=1, padding=1),
            BatchNorm2d(32),
            PReLU(),
            Conv2d(32, c, 3, stride=1, padding=1),
            Sigmoid()
        )

        self.oligo = Tooligo()

    def forward(self, enc_input, unit_num, seg_length):
        x = enc_input/255.0
        x = self.convs(x)
        x = reshape(x, (-1, unit_num))
        en = self.oligo(x)
        segment_x = reshape(en, (-1, seg_length))
        return en, segment_x


class Channel_M(nn.Module):
    def __int__(self):
        super(Channel_M, self).__int__()

    def forward(self, segment_en, sub_error, del_error, ins_error):
        # channel
        channel_x1 = sub_channel(segment_en, sub_error)
        channel_x1, length = del_channel(channel_x1, del_error)
        channel_x1, length2 = ins_channel(channel_x1, ins_error, length)

        channel_x2 = sub_channel(segment_en, sub_error)
        channel_x2, length = del_channel(channel_x2, del_error)
        channel_x2, length2 = ins_channel(channel_x2, ins_error, length)
        channel_x3 = sub_channel(segment_en, sub_error)
        channel_x3, length = del_channel(channel_x3, del_error)
        channel_x3, length2 = ins_channel(channel_x3, ins_error, length)
        channel_x4 = sub_channel(segment_en, sub_error)
        channel_x4, length = del_channel(channel_x4, del_error)
        channel_x4, length2 = ins_channel(channel_x4, ins_error, length)

        channel_x_all = torch.add(stack([channel_x1, channel_x2, channel_x3, channel_x4], dim=1), 1)

        return channel_x_all


def channel_test(segment_en, sub_error, del_error, ins_error):
    channel_x1 = sub_channel(segment_en, sub_error)
    channel_x1, length = del_channel(channel_x1, del_error)
    channel_x1, length2 = ins_channel(channel_x1, ins_error, length)
    channel_x2 = sub_channel(segment_en, sub_error)
    channel_x2, length = del_channel(channel_x2, del_error)
    channel_x2, length2 = ins_channel(channel_x2, ins_error, length)
    channel_x3 = sub_channel(segment_en, sub_error)
    channel_x3, length = del_channel(channel_x3, del_error)
    channel_x3, length2 = ins_channel(channel_x3, ins_error, length)
    channel_x4 = sub_channel(segment_en, sub_error)
    channel_x4, length = del_channel(channel_x4, del_error)
    channel_x4, length2 = ins_channel(channel_x4, ins_error, length)


    channel_x_all = torch.add(stack([channel_x1, channel_x2, channel_x3, channel_x4], dim=1), 1)

    return channel_x_all


class Decoder_M(nn.Module):
    def __init__(self, c):
        super(Decoder_M, self).__init__()

        self.Con_for_chl = Sequential(
            Conv1d(4, 1, 3, stride=1, padding=0),          # Lout = (Lin + 2*padding - kernel_size)/stride +1
            PReLU()
        )

        self.Dcovs = Sequential(
            ConvTranspose2d(c, 32, 3, stride=1, padding=1),
            PReLU(),
            ConvTranspose2d(32, 32, 3, stride=1, padding=1),
            BatchNorm2d(32),
            PReLU(),
            ConvTranspose2d(32, 32, 3, stride=1, padding=1),
            BatchNorm2d(32),
            PReLU(),
            ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            BatchNorm2d(16),
            PReLU(),
            ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            Sigmoid()
        )

    def forward(self, x, c):
        x = self.Con_for_chl(x).squeeze()
        x = reshape(x, (-1, c, 8, 8))
        x = self.Dcovs(x)
        x = x * 255.0
        return x


class Myround(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x = input*3.0
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Tooligo(nn.Module):
    def __init__(self):
        super(Tooligo, self).__init__()

    def forward(self, x):
        return Myround.apply(x)


def del_channel(x, error):
    mini_afterdel_all = torch.tensor([])
    len_afterdel_all = torch.tensor([])
    for mini in x:
        after_mini = mini
        del_patten = patten(mini, error)
        del_index = torch.nonzero(del_patten == 1).squeeze()
        while del_index.numel():
            if del_index.numel() > 1:
                e = del_index[0]
            else:
                e = del_index
            after_mini = cat((after_mini[0:e], after_mini[e + 1:]), dim=0)
            del_patten = cat((del_patten[0:e], del_patten[e + 1:]), dim=0)
            del_index = torch.nonzero(del_patten == 1).squeeze()
        len_afterdel = torch.Tensor([after_mini.shape[0]])
        mini_afterdel = cat((after_mini, (torch.ones((mini.shape[0]+2-after_mini.shape[0]))*-1.0)), dim=0).unsqueeze(0)
        mini_afterdel_all = cat((mini_afterdel_all, mini_afterdel), dim=0)
        len_afterdel_all = cat((len_afterdel_all, len_afterdel), dim=0)
    len_afterdel_all = reshape(len_afterdel_all, (-1, 1))

    return mini_afterdel_all, len_afterdel_all


def ins_channel(x, error, len_afterdel_all):
    mini_afterins_all = torch.tensor([])
    len_afterins_all = torch.tensor([])
    flag = 0
    for mini in x:
        after_mini = mini
        ins_patten = patten(after_mini[0:len_afterdel_all[flag].int()], error)
        ins_index = torch.nonzero(ins_patten == 1).squeeze()
        len_afterins = len_afterdel_all[flag] + sum(ins_patten)
        len_afterins_all = cat((len_afterins_all, len_afterins), dim=0)
        while ins_index.numel():
            if ins_index.numel() > 1:
                e = ins_index[0]
            else:
                e = ins_index
            after_mini = cat((after_mini[0:e+1], randint(0, 4, (1, )), after_mini[e + 1:]), dim=0)
            ins_patten = cat((ins_patten[0:e], torch.tensor([0, 0]), ins_patten[e + 1:]), dim=0)
            ins_index = torch.nonzero(ins_patten == 1).squeeze()
        flag = flag + 1
        mini_afterins = after_mini[0:mini.shape[0]].unsqueeze(0)
        mini_afterins_all = cat((mini_afterins_all, mini_afterins), dim=0)
    len_afterins_all = reshape(len_afterins_all, (-1, 1))
    return mini_afterins_all, len_afterins_all


def patten(x, error):

    temp = rand(x.shape)
    tem1 = torch.ones(x.shape)
    tem2 = torch.zeros(x.shape)
    tem3 = torch.where(temp < error, tem1, tem2)
    return tem3


def sub_channel(x, error):
    data = randint(0, 4, x.shape)
    sub_patten_pre = patten(x, error)
    sub_patten = mul(data, sub_patten_pre)
    x = add(x, sub_patten)
    x = remainder(x, 4)
    return x


def cul_para(rate_base):
    unit_num = int(1024 * rate_base)
    c = int(unit_num / 64)
    return unit_num, c


if __name__ == '__main__':

    all_error = 0.005
    sub_error = 0.43 * all_error
    del_error = 0.40 * all_error
    ins_error = 0.17 * all_error
    rate_base = 0.75

    (unit_num, c) = cul_para(rate_base)
    dna_encoder = Encoder_M(c=12)
    dna_decoder = Decoder_M(c=12)
    dna_channel = Channel_M()
    dna_djscc = DNA_djscc_M(dna_encoder, dna_channel, dna_decoder)
    input_images = torch.round(torch.rand((50, 3, 32, 32)) * 255.0)
    encoder_output, channel_en, decoder_output = dna_djscc(input_images, sub_error, del_error, ins_error, rate_base=rate_base)
    output1, output2 = dna_djscc.encoder(input_images, unit_num=unit_num, seg_length=256)
    output3 = channel_test(output2, sub_error, del_error, ins_error)
    output4 = dna_djscc.decoder(output3, c=c)
    print(output4.size())








