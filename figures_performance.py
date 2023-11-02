import numpy as np
import matplotlib.pyplot as plt
import torch

# data input

C1L1_psnr =[
    # r = 0.125  0.25    0.5     0.75       1       1.5
    [20.0682, 22.2669, 23.9882, 24.9695, 25.0621, 26.1312],  # ner=0   ************************24.8884
    [19.9222, 21.2611, 23.0331, 24.1713, 24.8995, 25.4415],  # ner=0.0025
    [19.7245, 20.7605, 22.6006, 23.1244, 23.6265, 24.5367],  # ner=0.005
    [18.9174, 20.2021, 21.8971, 22.7420, 23.2459, 24.4751],   # ner=0.0075
    [18.6303, 20.1855, 21.4557, 22.1266, 22.6909, 24.0371]    # ner=0.01
]
C2L1_psnr =[
    # r = 0.125  0.25    0.5     0.75      1       1.5
    [19.6695, 20.6587, 22.6235, 23.3835, 24.3862, 24.9893],  # ner=0
    [19.6096, 20.9986, 22.1836, 23.5381, 23.8757, 24.6557],  # ner=0.0025
    [18.9944, 20.2857, 22.2454, 22.6235, 23.6698, 24.4299],  # ner=0.005
    [18.9597, 20.5214, 22.0692, 22.5204, 23.5401, 24.1787],   # ner=0.0075
    [18.5442, 20.0818, 21.3343, 22.2783, 23.3278, 23.6697]    # ner=0.01
]
C1_homo = [
    [145.2966, 30.7819, 8.7260, 3.3028, 1.0981, 0.3710, 0.4077],   #r = 0.125   L1
    [186.1911, 19.8423, 6.0220, 1.5320, 0.5941, 0.2495, 0.3089],   #r = 0.5     L1
    [164.7075, 32.4679, 6.0260, 1.4232, 0.1946, 0.1877, 0.0795],   #r = 0.125   L2
    [174.8259, 30.5374, 4.6128, 0.9340, 0.3020, 0.1321, 0.0325],   #r = 0.5     L2
    [89.4432, 30.9885, 12.0839, 5.4212, 2.5165, 1.5066, 9.3077],   #r = 0.125  nomyloss
    [90.2859, 30.9486, 12.2261, 5.4995, 2.7338, 1.4620, 8.4845]    #r = 0.5  nomyloss
]

TEMP = np.array(C1_homo)
patten = np.array([1,2,3,4,5,6,0])
TEMP = TEMP*patten
TEMP[:,6]=(256-TEMP.sum(axis=1))
TEMP = TEMP/256
C1_homo2 = TEMP.tolist()


C1_agct = [
    [0.2083, 0.2878, 0.2622, 0.2417],   # 0.125  L1
    [0.2182, 0.2702, 0.2846, 0.2271],   # 0.5    L1
    [0.2333, 0.2548, 0.2627, 0.2492],   # 0.125  L2
    [0.2311, 0.2487, 0.2718, 0.2485],   # 0.5    L2
    [0.1622, 0.3014, 0.2952, 0.2412],   # r = 0.125  nomyloss
    [0.2417, 0.2753, 0.2950, 0.1880]    # r = 0.5  nomyloss
]

TEMP = np.array(C1_agct)
C1_agct2 = TEMP[:,1]+TEMP[:,2]
print(C1_agct2)
C1_agct2 = C1_agct2.tolist()
C1_agct2 = [float('{:.4f}'.format(i)) for i in C1_agct2]



def main(flag):
    linewidth = 1.5
    # rate_base---psnr
    R = [0.125, 0.25, 0.5, 0.75, 1, 1.5]
    r = [i/3 for i in R]
    if flag==0:
        # 设置画布大小像素点
        #plt.figure(figsize=(4, 4))
        #plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%,1%  r-panr 不同信道测试参数的r-psnr


        plt.plot(r, C1L1_psnr[2][:], linewidth=linewidth, linestyle='-.', marker='*', ms=5,
                 label="Proposed $\gamma_{train}=\gamma_{test}=0.50\%$")
        plt.plot(r, C2L1_psnr[4][:], linewidth=linewidth, linestyle='-', marker='^', ms=5,
                 label="Proposed $\gamma_{train}=\gamma_{test}=1.00\%$")
        plt.plot(r, C1L1_psnr[0][:], linewidth=linewidth, linestyle='-.', marker='*', ms=5,
                 label="Proposed $\gamma_{train}=0.50\%$, $\gamma_{test}=0.00\%$")
        plt.plot(r, C2L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='*', ms=5,
                 label="Proposed $\gamma_{train}=1.00\%$, $\gamma_{test}=0.00\%$")


        base_rate_VAEU = [0.046875, 0.09375, 0.1406, 0.1875, 0.425, 0.46875]
        r3 = [i / 3 for i in base_rate_VAEU]
        Psnr_VAEN = [12.9182, 12.8494, 12.7280, 12.8859, 12.5672, 12.6334]
        plt.plot(r3, Psnr_VAEN, linewidth=linewidth, linestyle='-', marker='*', ms=5,
                 label="VAE+QC, $\gamma_{test}=0.50\%$")


        base_rate_VAEU2 = [0.42, 0.3, 0.16, 0.12, 0.1, 0.06, 0.025]
        r2 = [i/3 for i in base_rate_VAEU2]
        Psnr_VAEN2 = [18.5, 18.5, 17.6, 17, 16.5, 14, 12]
        plt.plot(r2, Psnr_VAEN2, linewidth=linewidth, linestyle='--', marker='s', ms=5,
                 label="VAEU")


        plt.xlabel('r(nts/pixel)', fontsize=12)
        plt.ylabel('PSNR', fontsize=12)
        # plt.title('z')
        plt.grid()
        #plt.legend(loc=4)
        plt.legend()  # bbox_to_anchor = (1.05, 1), loc=2, borderaxespad=0
        plt.show()
    elif flag == 1:
        #plt.figure(figsize=(4, 4))
        #plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        plt.plot(r, C1L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='*', ms=5,
                 label="$\gamma_{test}=0.00\%$")
        plt.plot(r, C1L1_psnr[1][:], linewidth=linewidth, linestyle='-', marker='^', ms=5,
                 label="$\gamma_{test}=0.25\%$")
        plt.plot(r, C1L1_psnr[2][:], linewidth=linewidth, linestyle='-', marker='h', ms=5,
                 label="$\gamma_{test}=0.50\%$")
        plt.plot(r, C1L1_psnr[3][:], linewidth=linewidth, linestyle='-', marker='s', ms=5,
                 label="$\gamma_{test}=0.75\%$")
        plt.plot(r, C1L1_psnr[4][:], linewidth=linewidth, linestyle='-', marker='D', ms=5,
                 label="$\gamma_{test}=1.00\%$")
        plt.xlabel('r(nts/pixel)', fontsize=12)
        plt.ylabel('PSNR', fontsize=12)
        plt.title('$\gamma_{train}=0.5\%$')
        plt.grid()
        plt.legend(loc=4)
        plt.show()
    elif flag == 2:
       # plt.figure(figsize=(4, 4))
        #plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        plt.plot(r, C2L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='*', ms=5,
                 label="$\gamma_{test}=0.00\%$")
        plt.plot(r, C2L1_psnr[1][:], linewidth=linewidth, linestyle='-', marker='^', ms=5,
                 label="$\gamma_{test}=0.25\%$")
        plt.plot(r, C2L1_psnr[2][:], linewidth=linewidth, linestyle='-', marker='h', ms=5,
                 label="$\gamma_{test}=0.50\%$")
        plt.plot(r, C2L1_psnr[3][:], linewidth=linewidth, linestyle='-', marker='s', ms=5,
                 label="$\gamma_{test}=0.75\%$")
        plt.plot(r, C2L1_psnr[4][:], linewidth=linewidth, linestyle='-', marker='D', ms=5,
                 label="$\gamma_{test}=1.00\%$")
        plt.xlabel('r(nts/pixel)', fontsize=12)
        plt.ylabel('PSNR', fontsize=12)
        plt.title('$\gamma_{train}=1.00\%$')
        plt.grid()
        plt.legend(loc=4)
        plt.show()
    elif flag == 3:
        #plt.figure(figsize=(8, 4))
        #plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        labels = ['1', '2', '3', '4', '5', '6', '7+']
        x = np.arange(len(labels))
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, C1_homo[0][:], width / 6, label='Proposed: b=1,  R=0.125')
        plt.bar(x - width / 3, C1_homo[1][:], width / 6, label='Proposed: b=1,  R=0.500')
        plt.bar(x - width / 6, C1_homo[2][:], width / 6, label='Proposed: b=10, R=0.125')
        plt.bar(x            , C1_homo[3][:], width / 6, label='Proposed: b=10, R=0.500')
        plt.bar(x + width / 6, C1_homo[4][:], width / 6, label='CNN+Q: R=0.125')
        plt.bar(x + width / 3, C1_homo[5][:], width / 6, label='CNN+Q: R=0.500')
        plt.xlabel('homologous length', fontsize=12)
        plt.xticks(x, ('1', '2', '3', '4', '5', '6', '7+'))
        plt.ylabel('Number per 256nt', fontsize=12)
        plt.ylim((0, 200))
        plt.grid()
        plt.legend()
        plt.show()
    elif flag == 4:
        #plt.figure(figsize=(6, 4))
       # plt.rcParams.update({"font.size": 9})
        #plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        labels = ['1', '2', '3', '4', '5', '6', '7+']
        x = np.arange(len(labels))
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, C1_homo2[0][:], width / 6, color='lightblue', label='Proposed: $G_1$, R=0.125')
        plt.bar(x - width / 3, C1_homo2[1][:], width / 6, color='lightcoral', label='Proposed: $G_1$, R=0.500')
        plt.bar(x - width / 6, C1_homo2[2][:], width / 6, color='lightskyblue', label='Proposed: $G_2$, R=0.125') # label='$y = %ix$' % i
        plt.bar(x, C1_homo2[3][:], width / 6, color='khaki', label='Proposed: $G_2$, R=0.500')
        plt.bar(x + width / 6, C1_homo2[4][:], width / 6, color='lightpink', label='CNN+Q: R=0.125')
        plt.bar(x + width / 3, C1_homo2[5][:], width / 6, color='lightseagreen', label='CNN+Q: R=0.500')
        plt.xlabel('portation of homologous length', fontsize=12)
        plt.xticks(x, ('1', '2', '3', '4', '5', '6', '7+'))
        plt.ylabel('portation(%)', fontsize=12)
        plt.ylim((0, 0.8))
        plt.grid()
        plt.legend()
        plt.show()
    elif flag == 5:
        plt.figure(figsize=(6, 6.5))
        plt.rcParams.update({"font.size": 9})
        plt.subplots_adjust(wspace=0.4, hspace=1)
        # \gamma=0.5%, r-psnr
        labels = ['A', 'C', 'G', 'T']
        x = np.arange(len(labels))
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, C1_agct[0][:], width / 6, label='Proposed: $G_1$, R=0.125')
        plt.bar(x - width / 3, C1_agct[1][:], width / 6, label='Proposed: $G_1$, R=0.500')
        plt.bar(x - width / 6, C1_agct[2][:], width / 6, label='Proposed: $G_2$, R=0.125')
        plt.bar(x, C1_agct[3][:], width / 6, label='Proposed: $G_2$, R=0.500')
        plt.bar(x + width / 6, C1_agct[4][:], width / 6, label='CNN+Q: R=0.125')
        plt.bar(x + width / 3, C1_agct[5][:], width / 6, label='CNN+Q: R=0.500')
        plt.xlabel('Nucleobase type', fontsize=12)
        plt.xticks(x, ('A', 'C', 'G', 'T'))
        plt.ylabel('portation(%)', fontsize=12)
        plt.ylim((0, 0.4))
        plt.grid()
        plt.legend()
        plt.show()
    elif flag == 6:
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        x = np.arange(len(labels))
        y = torch.tensor(C1_agct)
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, y[:, 0], width / 4, color='bisque', label='A')
        plt.bar(x - width / 4, y[:, 1], width / 4, color='lightcoral', label='C')
        plt.bar(x            , y[:, 2], width / 4, color='lightseagreen', label='G')
        plt.bar(x + width / 4, y[:, 3], width / 4, color='lightsteelblue', label='T')
        plt.xlabel('method', fontsize=12)
        plt.xticks(x, ('a', 'b', 'c', 'd', 'e', 'f'))
        plt.ylabel('portation(%)', fontsize=12)
        plt.ylim((0, 0.5))
        plt.grid()
        plt.legend(loc=2)
        ax2 = plt.twinx()
        ax2.set_ylabel("portation(%)", fontsize=12)
        ax2.set_ylim([0.35, 0.65])
        plt.plot(x, C1_agct2, color='r', marker='.', ms=5, linewidth='1', label="GC-content")
        # 显示数字
        for a, b in zip(x, C1_agct2):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
        # 在右侧显示图例
        plt.legend(loc=1)
        plt.show()



if __name__ == '__main__':
    #main(0)
    main(1)
    main(2)
    main(4)
    main(6)

'''
linewidth = 1.5
plt.subplot(1, 3, 1)
base_rate_VAEU = [0.046875, 0.09375, 0.1406, 0.1875, 0.425, 0.46875]
Psnr_VAEN = [12.9182, 12.8494, 12.7280, 12.8859, 12.5672, 12.6334]
#base_rate_VAEU2 = [0.42, 0.3, 0.16, 0.12, 0.1, 0.06, 0.025]
#Psnr_VAEN2 = [18.5, 18.5, 17.6, 17, 16.5, 14, 12]
base_rate_DJSCC = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
Psnr_DJSCC = [19.4387, 21.0124, 22.5855, 23.5173, 24.0494, 24.5375, 25.0631, 25.0239]
plt.plot(base_rate_VAEU, Psnr_VAEN, color='#0C84C6', linewidth=linewidth, linestyle='-', marker='*', ms=5, label="VAE+QC")
plt.plot(base_rate_DJSCC, Psnr_DJSCC, color='#F44D4D', linewidth=linewidth, linestyle='-', marker='*',  label="Proposed")
plt.xlabel('base rate (base per pixel)',fontsize=12)
plt.ylabel('PSNR',fontsize=12)
plt.grid()
plt.legend()
# semilogx

plt.subplot(1, 3, 2)
labels = ['1', '2', '3', '4', '5', '6', '7+']
homologous_CNN = [89.4432,30.9885,12.0839,5.4212,2.5165,1.5066,9.3077]   # r = 0.125
homologous_CNN2 = [89.4449,30.9849,12.0854,5.4229,2.5157,1.5065,9.3086]  # r = 0.5
# acgt_rate_CNN2 = [0.1310, 0.2368, 0.2840, 0.3482]   # r = 0.46875
homologous_DJSCC = [145.2966,30.7819,8.7260,3.3028,1.0981,0.3710,0.4077]  # base_rate = 0.125
homologous_DJSCC2 = [161.9930,32.1287,5.9576,1.8300,0.2688,0.2440,0.2746]   # base_rate = 1.75
x = np.arange(len(labels))
width = 0.8  # the width of the bars
plt.bar(x - width/4, homologous_CNN,   width/4, color='#FFA510', label='CNN+Map:  r=0.125')
plt.bar(x , homologous_CNN2,  width/4, color='#FFBD66', label='CNN+Map:  r=0.5')
plt.bar(x + width/4, homologous_DJSCC,  width/4, color='#2455A4', label='Proposed: a=200,b=1,  r=0.125')
plt.bar(x + width/2, homologous_DJSCC2, width/4, color='#F44D4D', label='Proposed: a=200,b=10,r=0.125')
plt.xlabel('homologous length', fontsize=12)
plt.xticks(x, ('1', '2', '3', '4', '5', '6', '7+'))
plt.ylabel('Number per 256nt', fontsize=12)
plt.ylim((0, 170))
plt.grid()
plt.legend()


plt.subplot(1, 3, 3)
labels = ['A', 'C', 'G', 'T']
acgt_rate_CNN = [0.1878, 0.3048, 0.2634, 0.2439]   # r = 0.125
acgt_rate_CNN2 = [0.1622, 0.3014, 0.2952, 0.2412]   # r = 0.5
# acgt_rate_CNN2 = [0.1310, 0.2368, 0.2840, 0.3482]   # r = 0.46875
acgt_rate_DJSCC = [0.2083, 0.2878, 0.2622, 0.2417]   # r = 0.125
acgt_rate_DJSCC2 = [0.2458, 0.2630, 0.2564, 0.2348]   # r = 1.5
x = np.arange(len(labels))
width = 0.6  # the width of the bars
plt.bar(x - width/4, acgt_rate_CNN,  width/4, color='#FFA510', label='CNN+Map: r=0.125')
plt.bar(x , acgt_rate_CNN2,  width/4, color='#FFBD66', label='CNN+Map: r=0.125')
plt.bar(x + width/4, acgt_rate_DJSCC,   width/4, color='#2455A4', label='Proposed: a=200,b=1,  r=0.125')
plt.bar(x + width/2, acgt_rate_DJSCC2,  width/4, color='#F44D4D', label='Proposed: a=200,b=10,r=0.125')
plt.xlabel('Base type', fontsize=12)
plt.xticks(x, ('A','C','G','T'))
plt.ylabel('AGCT rate', fontsize=12)
plt.ylim((0, 0.5))
plt.grid()
plt.legend()
plt.show()

'''
# plt.show()

'''


plt.subplot(1, 3, 2)
x = np.linspace(1, 7, 7, endpoint=True)
homologous_VAEU = [173.5265, 26.4022, 5.6221, 1.1551, 0.4978, 0.1129, 0.1586]   # base_rate = 0.046875
homologous_VAEU2 = [166.9021, 20.9815, 7.1503, 2.1223, 0.7640, 0.1599, 1.9082]   # base_rate = 0.46875
# homologous_DJSCC = [169.7485, 27.5215, 6.8167, 1.8274, 0.3452, 0.1552, 0.1312]   # base_rate = 0.125
homologous_DJSCC = [145.2966,30.7819,8.7260,3.3028,1.0981,0.3710,0.4077]  # base_rate = 0.125
homologous_DJSCC2 = [180.8770, 22.0992, 6.1673, 1.7425, 0.6325, 0.2178, 0.1499]   # base_rate = 1.75
plt.plot(x, homologous_VAEU, color='#0C84C6', linewidth=linewidth, linestyle='-', marker='*', ms=5, label="VAE+QC,  r=0.047")
plt.plot(x, homologous_VAEU2, color='#0C84C6', linewidth=linewidth, linestyle='--', marker='*', ms=5, label="VAE+QC,  r=0.468")
plt.plot(x, homologous_DJSCC, color='#F44D4D', linewidth=linewidth, linestyle='-', marker='*', ms=5, label="Proposed, r=0.125")
plt.plot(x, homologous_DJSCC2, color='#F44D4D', linewidth=linewidth, linestyle='--', marker='*', ms=5, label="Proposed, r=1.750")
plt.xlabel('Homologous length',fontsize=12)
plt.xticks(x, ('1','2','3','4','5','6','7+'))
plt.ylabel('Average number',fontsize=12)
plt.grid()
plt.legend()
'''