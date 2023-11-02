import numpy as np
import matplotlib.pyplot as plt
import torch

# data_50peoch input

C1L1_psnr =[
    # r = 0.125  0.25    0.5     0.75       1       1.5
    [20.0682, 22.2669, 23.9882, 24.9695, 25.3285, 26.1312],  # ner=0   ************************24.8884
    [19.9222, 21.2611, 23.1537, 24.1713, 24.8995, 25.4415],  # ner=0.0025
    [19.7245, 20.7605, 22.6006, 23.6112, 23.6265, 24.7974],  # ner=0.005
    [18.9174, 20.2021, 21.8971, 22.7420, 23.2459, 24.4751],   # ner=0.0075
    [18.6303, 20.1855, 21.4557, 22.1266, 22.6909, 24.0371],    # ner=0.01
    [17.6146, 19.6345, 20.3273, 21.6527, 22.5102, 23.2764]   # ner=0.0125
]
C2L1_psnr =[
    # r = 0.125  0.25    0.5     0.75      1       1.5
    [19.6695, 20.6587, 22.6235, 23.4974, 24.3862, 24.9893],  # ner=0
    [19.6096, 20.9986, 22.4350, 23.5381, 23.8757, 24.6557],  # ner=0.0025
    [18.9944, 20.2857, 22.2454, 22.6235, 23.6698, 24.4299],  # ner=0.005
    [18.9597, 20.5214, 22.0692, 22.5204, 23.5401, 24.1787],   # ner=0.0075
    [18.5442, 20.0818, 21.3343, 22.2783, 23.3278, 23.6697],    # ner=0.01
    [19.0742, 19.8395, 20.3429, 22.1416, 23.0327, 23.8090]   # ner=0.0125
]
C1_homo = [
    [145.2966, 30.7819, 8.7260, 3.3028, 1.0981, 0.3710, 0.4077],   #r = 0.125   L1
    [186.1911, 19.8423, 6.0220, 1.5320, 0.5941, 0.2495, 0.3089],   #r = 0.5     L1
    [164.7075, 32.4679, 6.0260, 1.4232, 0.1946, 0.1877, 0.0795],   #r = 0.125   L2
    [174.8259, 30.5374, 4.6128, 0.9340, 0.3020, 0.1321, 0.0325],   #r = 0.5     L2
    [89.4432, 30.9885, 12.0839, 5.4212, 2.5165, 1.5066, 9.3077],   #r = 0.125  nomyloss
    [90.2859, 30.9486, 12.2261, 5.4995, 2.7338, 1.4620, 8.4845]    #r = 0.5  nomyloss
]

C1_multi = [
    # 0,  0.0025  0.005    0.0075   0.01
    [20.2369, 20.1872, 19.8481, 19.2682, 18.7917, 17.8207],    #r = 0.125
    [23.8427, 23.7699, 23.4216, 22.5952, 22.1739, 21.6701]     #r = 0.5
]

TEMP = np.array(C1_homo)
patten = np.array([1,2,3,4,5,6,0])
TEMP = TEMP*patten
TEMP[:,6]=(256-TEMP.sum(axis=1))
TEMP = TEMP/256*100
C1_homo2 = TEMP.tolist()
print(C1_homo2)

C1_agct = [
    [0.2083, 0.2878, 0.2622, 0.2417],   # 0.125  L1
    [0.2182, 0.2702, 0.2846, 0.2271],   # 0.5    L1
    [0.2333, 0.2548, 0.2627, 0.2492],   # 0.125  L2
    [0.2311, 0.2487, 0.2718, 0.2485],   # 0.5    L2
    [0.1622, 0.3014, 0.2952, 0.2412],   # r = 0.125  nomyloss
    [0.2417, 0.2753, 0.2950, 0.1880]    # r = 0.5  nomyloss
]

TEMP = np.array(C1_agct)
C1_agct2 = (TEMP[:,1]+TEMP[:,2])*100
print(C1_agct2)
C1_agct2 = C1_agct2.tolist()
C1_agct2 = [float('{:.4f}'.format(i)) for i in C1_agct2]



def main(flag):
    linewidth = 1.5
    ms=6
    fontsize = 10
    # rate_base---psnr
    R = [0.125, 0.25, 0.5, 0.75, 1, 1.5]
    #    1/24   1/12  1/6  1/4  1/3 1/2
    r = [i/3 for i in R]
    if flag==0:
        # 设置画布大小像素点
        # \gamma=0.5%,1%  r-panr 不同信道测试参数的r-psnr
        z_dim_VAEU = [4,   6,     12,    24,   32,    40]
        #            1/64  3/128  3/64  3/32   1/8    5/32
        r3 = [i*4/32/32 for i in z_dim_VAEU]
        Psnr_VAEN = [12.9182, 12.8844, 12.7280, 12.5277, 12.5672, 12.6334]
        Psnr_VAEN2 = [15.2058, 16.1213, 17.2873, 17.6499, 17.8603, 18.3961]



#        plt.plot(r, C2L1_psnr[4][:], linewidth=linewidth, linestyle='-', marker='^', ms=ms,
#                 label="Proposed $\gamma_{train}=\gamma_{test}=1.00\%$")
#        plt.plot(r, C2L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='^', ms=ms,
#                 label="Proposed $\gamma_{train}=1.00\%$, $\gamma_{test}=0.00\%$")


        plt.semilogx(r, C1L1_psnr[2][:], 'b', linewidth=linewidth, linestyle='-', marker='o', ms=ms,
                 label="Proposed: $\gamma_{test}=0.50\%$")
        plt.semilogx(r, C1L1_psnr[0][:], 'b',linewidth=linewidth, linestyle='-', marker='s', ms=ms,
                 label="Proposed: $\gamma_{test}=0.00\%$")
        plt.semilogx(r3, Psnr_VAEN, 'r',linewidth=linewidth, linestyle='-.', marker='o', ms=ms,
                 label="VAEU-QC: $\gamma_{test}=0.50\%$")
        plt.semilogx(r3, Psnr_VAEN2, 'r',linewidth=linewidth, linestyle='-.', marker='s', ms=ms,
                 label="VAEU-QC: $\gamma_{test}=0.00\%$")

        plt.xlabel('R(nts/pixel)', fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        plt.grid()
        #plt.legend(loc=4)
        plt.legend()  # bbox_to_anchor = (1.05, 1), loc=2, borderaxespad=0
        plt.show()
    elif flag == 1:
        # \gamma=0.5%, r-psnr
        #plt.plot(r, C1L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='*', ms=ms,
        #         label="$\gamma_{test}=0.00\%$")
        plt.plot(r, C1L1_psnr[1][:], 'b', linewidth=linewidth, linestyle='-', marker='^', ms=ms,
                 label="$\gamma_{test}=0.25\%$")
        plt.plot(r, C1L1_psnr[2][:], 'r', linewidth=linewidth, linestyle='-', marker='h', ms=ms,
                 label="$\gamma_{test}=0.50\%$")
        plt.plot(r, C1L1_psnr[3][:], 'g', linewidth=linewidth, linestyle='-', marker='s', ms=ms,
                 label="$\gamma_{test}=0.75\%$")
        plt.plot(r, C1L1_psnr[4][:], 'y', linewidth=linewidth, linestyle='-', marker='D', ms=ms,
                 label="$\gamma_{test}=1.00\%$")
        plt.plot(r, C1L1_psnr[5][:], 'c', linewidth=linewidth, linestyle='-', marker='o', ms=ms,
                 label="$\gamma_{test}=1.25\%$")
        plt.xlabel('R(nts/pixel)', fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        plt.title('$\gamma_{train}=0.5\%$', fontsize=fontsize)
        plt.ylim([17, 26])
        plt.grid()
        plt.legend(loc=4)
        plt.show()
    elif flag == 2:
       # plt.figure(figsize=(4, 4))
        #plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        #plt.plot(r, C2L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='*', ms=ms,
        #         label="$\gamma_{test}=0.00\%$")
        plt.plot(r, C2L1_psnr[1][:], 'b', linewidth=linewidth, linestyle='-', marker='^', ms=ms,
                 label="$\gamma_{test}=0.25\%$")
        plt.plot(r, C2L1_psnr[2][:], 'r', linewidth=linewidth, linestyle='-', marker='h', ms=ms,
                 label="$\gamma_{test}=0.50\%$")
        plt.plot(r, C2L1_psnr[3][:], 'g', linewidth=linewidth, linestyle='-', marker='s', ms=ms,
                 label="$\gamma_{test}=0.75\%$")
        plt.plot(r, C2L1_psnr[4][:], 'y', linewidth=linewidth, linestyle='-', marker='D', ms=ms,
                 label="$\gamma_{test}=1.00\%$")
        plt.plot(r, C2L1_psnr[5][:], 'c', linewidth=linewidth, linestyle='-', marker='o', ms=ms,
                 label="$\gamma_{test}=1.25\%$")
        plt.xlabel('R(nts/pixel)', fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        plt.ylim([17, 26])
        plt.title('$\gamma_{train}=1.00\%$', fontsize=fontsize)
        plt.grid()
        plt.legend(loc=4)
        plt.show()
    elif flag == 3:
        #plt.figure(figsize=(6, 4))
       # plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)   1/24   1/12  1/6  1/4  1/3 1/2
        # \gamma=0.5%, r-psnr
        labels = ['1', '2', '3', '4', '5', '6', '7+']
        x = np.arange(len(labels))
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, C1_homo2[0][:], width / 6,  label='Proposed: $G_1$, R=1/24')
        plt.bar(x - width / 3, C1_homo2[1][:], width / 6,  label='Proposed: $G_1$, R=1/6')
        plt.bar(x - width / 6, C1_homo2[2][:], width / 6,  label='Proposed: $G_2$, R=1/24') # label='$y = %ix$' % i
        plt.bar(x, C1_homo2[3][:], width / 6,  label='Proposed: $G_2$, R=1/6')
        plt.bar(x + width / 6, C1_homo2[4][:], width / 6,  label='CNN-M: R=1/24')
        plt.bar(x + width / 3, C1_homo2[5][:], width / 6,  label='CNN-M: R=1/6')
        plt.xlabel('Proportion of homologous run length', fontsize=fontsize)
        plt.xticks(x, ('1', '2', '3', '4', '5', '6', '7+'))
        plt.ylabel('Proportion(%)', fontsize=fontsize)
        plt.ylim((0, 80))
        plt.grid()
        plt.legend()
        plt.show()
    elif flag == 6:
        labels = ['5', '6', '7+']
        x = np.arange(len(labels))
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, C1_homo2[0][4:7], width / 6, color='lightblue')
        plt.bar(x - width / 3, C1_homo2[1][4:7], width / 6, color='lightcoral')
        plt.bar(x - width / 6, C1_homo2[2][4:7], width / 6, color='lightskyblue')  # label='$y = %ix$' % i
        plt.bar(x, C1_homo2[3][4:7], width / 6, color='khaki')
        plt.bar(x + width / 6, C1_homo2[4][4:7], width / 6, color='lightpink')
        plt.bar(x + width / 3, C1_homo2[5][4:7], width / 6, color='lightseagreen')
       # plt.xlabel('Proportion of homologous run length', fontsize=fontsize)
        plt.xticks(x, ('5', '6', '7+'))
        plt.ylabel('Proportion(%)', fontsize=fontsize)
        plt.ylim((0, 10))
        plt.grid()
        plt.legend()
        plt.show()
    elif flag == 4:
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        # C1_agct = [i*100 for i in C1_agct]
        x = np.arange(len(labels))
        y = torch.tensor(C1_agct)*100
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, y[:, 0], width / 4, color='bisque', label='A')
        plt.bar(x - width / 4, y[:, 1], width / 4, color='lightcoral', label='C')
        plt.bar(x            , y[:, 2], width / 4, color='lightseagreen', label='G')
        plt.bar(x + width / 4, y[:, 3], width / 4, color='lightsteelblue', label='T')
        plt.xlabel('Method', fontsize=fontsize)
        plt.xticks(x, ('a', 'b', 'c', 'd', 'e', 'f'))
        plt.ylabel('Proportion(%)', fontsize=fontsize)
        plt.ylim((0, 50))
        plt.grid()
        plt.legend(loc=2)
        ax2 = plt.twinx()
        ax2.set_ylabel("Proportion(%)", fontsize=fontsize)
        ax2.set_ylim([35, 65])
        plt.plot(x, C1_agct2, color='r', marker='.', ms=5, linewidth='1', label="GC-content")
        # 显示数字
        for a, b in zip(x, C1_agct2):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=fontsize)
        # 在右侧显示图例
        plt.legend(loc=1)
        plt.show()
    elif flag == 5:

        y = [0, 0.25, 0.5, 0.75, 1, 1.25]
        v2 = torch.tensor(C1L1_psnr)
        plt.plot(y, C1_multi[0][:], 'b', linewidth=linewidth, linestyle='-', marker='o', ms=ms,
                 label="$v=4, R=1/24$")
        plt.plot(y, C1_multi[1][:],  'b', linewidth=linewidth, linestyle='-', marker='s', ms=ms,
                 label="$v=4, R=1/6$")
        plt.plot(y, v2[:,0],  'r', linewidth=linewidth, linestyle='-.', marker='o', ms=ms,
                 label="$v=2, R=1/24$")
        plt.plot(y, v2[:,2],  'r', linewidth=linewidth, linestyle='-.', marker='s', ms=ms,
                 label="$v=2, R=1/6$")
        plt.xlabel('$\gamma_{test}(\%)$', fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        # plt.title('$\gamma_{train}=1.00\%$', fontsize=fontsize)
        plt.grid()
        plt.legend(loc=1)
        plt.show()



if __name__ == '__main__':
    for i in range(4, 7):
        main(i)
    #main(5)

