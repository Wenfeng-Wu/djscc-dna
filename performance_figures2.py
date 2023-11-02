import numpy as np
import matplotlib.pyplot as plt
import torch

# data_50peoch input
L = [25, 50, 75, 100, 125, 150, 175, 200]
r0125_L_psnr =[19.3887, 19.1885, 19.2354, 19.1047, 19.1301, 19.2673, 18.9751, 18.9549]
r05_L_psnr = [22.5290, 22.3978, 22.3341, 22.1627, 22.0755,21.8939 , 21.9565, 21.8670]

r0125_L_ssim = [0.5817, 0.5659, 0.5642, 0.5617, 0.5588, 0.5636, 0.5499, 0.5447]
r05_L_ssim = [0.7680, 0.7629, 0.7558, 0.7509, 0.7452, 0.7327, 0.7371, 0.7319]

r0125_L_rll = [[141.8577423095703, 30.30658531188965 , 9.852261543273926 , 3.0095107555389404 , 1.1701565980911255 , 0.4539242386817932 , 0.6049792170524597 ],
               [147.85784912109375, 29.846826553344727 , 8.93474292755127 , 3.055786371231079 , 1.0075926780700684 , 0.4058104157447815 , 0.3334398865699768 ],
                [163.7069549560547, 26.99376678466797 , 8.732536315917969 , 1.4067295789718628 , 0.7377317547798157 , 0.22094789147377014 , 0.21671195328235626 ],
                [147.10426330566406, 32.461517333984375 , 8.83020305633545 , 2.3708839416503906 , 0.8118606209754944 , 0.2858855426311493 , 0.33072251081466675 ],
                [167.80287170410156 , 26.5401611328125 , 8.432984352111816 , 1.1907368898391724 , 0.6036604642868042 , 0.13810741901397705 , 0.1819852888584137 ],
                [164.38363647460938, 27.674911499023438 , 8.228859901428223 , 1.4536445140838623 , 0.7551550269126892 , 0.125 , 0.17790921032428741],
                [164.98129272460938 , 33.41000747680664 , 4.951167106628418 , 1.4732656478881836 , 0.3132193088531494 , 0.12859654426574707 , 0.16759911179542542 ],
                [167.64578247070312, 31.468549728393555 , 5.101062774658203 , 1.843829870223999 , 0.23589354753494263 , 0.09990409016609192 , 0.13802748918533325 ]

]

r05_L_rll = [[ 142.8656463623047 , 30.749679565429688 , 8.660965919494629 , 3.4314358234405518 , 1.2037644386291504 , 0.33550789952278137 , 0.6656309962272644 ],  #25
                [ 162.74432373046875 , 25.197490692138672 , 8.498421669006348 , 2.4124341011047363 , 0.8795256614685059 , 0.22016863524913788 , 0.3229799270629883 ], #50
                [ 180.47903442382812 , 22.860244750976562 , 5.818214416503906 , 1.6092650890350342 , 0.6468490362167358 , 0.17088595032691956 , 0.253296822309494 ], #75
                [170.14138793945312, 27.13984489440918, 6.6513848304748535, 1.5887048244476318, 0.6066576242446899, 0.11036404967308044, 0.22798113524913788],     #100
                [ 171.02777099609375 , 29.37246322631836 , 5.0736494064331055 , 1.5150655508041382 , 0.5147957801818848 , 0.1341012567281723 , 0.23560382425785065 ], #125
                [ 166.84812927246094 , 24.876298904418945 , 7.83233118057251 , 2.058563709259033 , 1.159956455230713 , 0.12522977590560913 , 0.16492167115211487 ], #150
                [ 160.7926788330078 , 30.955142974853516 , 6.86476993560791 , 2.366128444671631 , 0.3382352888584137 , 0.09905491024255753 , 0.14959639310836792 ], #175
                [ 175.23458862304688 , 27.879436492919922 , 5.399106979370117 , 1.5688538551330566 , 0.24329642951488495 , 0.12750759720802307 , 0.08313019573688507 ] #200
]

r0125_L_gc = [[0.2021484375, 0.2734375, 0.2646484375, 0.259765625],
              [0.232421875, 0.25830078125, 0.25341796875, 0.255859375],
              [0.2294921875, 0.24609375, 0.25927734375, 0.26513671875],
              [0.234375, 0.248046875, 0.25390625, 0.263671875],
              [0.21337890625, 0.251953125, 0.27001953125, 0.2646484375],
              [0.24267578125, 0.2451171875, 0.267578125, 0.24462890625],
              [0.228515625, 0.27587890625, 0.2548828125, 0.24072265625],
              [0.205078125, 0.26171875, 0.2646484375, 0.2685546875]

]

r05_L_gc = [[0.2364501953125, 0.2633056640625, 0.2418212890625, 0.2584228515625],    #25
                [0.2357177734375, 0.2550048828125, 0.2607421875, 0.24853515625],     #50
                [0.24609375, 0.26171875, 0.2554931640625, 0.2366943359375],          #75
                [0.2503662109375, 0.2598876953125, 0.2608642578125, 0.2288818359375],#100
                [0.2178955078125, 0.2349853515625, 0.284423828125, 0.2626953125],    #125
                [0.23095703125, 0.254638671875, 0.27392578125, 0.240478515625],      #150
                [0.23876953125, 0.255126953125, 0.2681884765625, 0.2379150390625],   #175
                [0.222900390625, 0.2509765625, 0.281005859375, 0.2451171875]         #200
]


C1_psnr = [
    # 0.125    0.25      0.5    0.75      1       1.25       1.5
    [19.2354, 20.7129, 22.3341, 23.0584, 23.8818, 24.3412, 24.5692],    # a=75
    [20.1358, 21.7420, 23.5335, 24.0330, 24.8374, 25.3682, 25.6183],  # a=75 y=0
    [18.9831, 20.5037, 21.9522, 22.7541, 23.3239, 23.8692, 24.2692],     # a=175
    [19.7939, 21.2795, 22.9570, 23.7162, 24.8374, 24.7844, 24.9332]   # a=175, y=0
]


#75


C1_ssim = [
    # 0.125    0.25      0.5    0.75     1       1.25     1.5
    [0.5640, 0.6681, 0.7452, 0.7886, 0.8278, 0.8418, 0.8509],    # a=75   y=0.005
    [0.6150, 0.7168, 0.8079, 0.8283, 0.8590, 0.8726, 0.8823],  # a=75 y=0
    [0.5499, 0.6503, 0.7367, 0.7804, 0.8044, 0.8227, 0.8509],     # a=175   y=0.005
    [0.5997, 0.6915, 0.7842, 0.8187, 0.8334, 0.8526, 0.8614]   # a=175, y=0
]



def cul_per_RLL(x):
    TEMP = np.array(x)
    patten = np.array([1, 2, 3, 4, 5, 6, 0])
    TEMP = TEMP * patten
    TEMP[:, 6] = (256 - TEMP.sum(axis=1))
    TEMP = TEMP / 256 * 100
    x3 = (TEMP[:, -2] + TEMP[:, -1])
    x2 = TEMP.tolist()  # 百分比
    x3 = x3.tolist()  # 后两个的百分比
    return x2, x3


def cul_gc(x):
    TEMP = np.array(x)
    x2 = (TEMP[:, 1] + TEMP[:, 2]) * 100
    x2 = x2.tolist()  # GC-content
    x2 = [float('{:.4f}'.format(i)) for i in x2]
    return x2


r0125_L_rll2, r0125_L_rll3 = cul_per_RLL(r0125_L_rll)
r05_L_rll2, r05_L_rll3 = cul_per_RLL(r05_L_rll)

r0125_L_gc2 = cul_gc(r0125_L_gc)
r05_L_gc2 = cul_gc(r05_L_gc)

GC = [  [0.2294921875, 0.24609375, 0.25927734375, 0.26513671875],  #Proposed-1:R=1/24
        [0.24609375, 0.26171875, 0.2554931640625, 0.2366943359375],  # Proposed-1:R=1/6
        [0.228515625, 0.27587890625, 0.2548828125, 0.24072265625],  # Proposed-2:R=1/24
        [0.23876953125, 0.255126953125, 0.2681884765625, 0.2379150390625],   # Proposed-2:R=1/6
        [0.1622, 0.3014, 0.2952, 0.2412],   # CNN-M:R=1/24
        [0.2417, 0.2753, 0.2950, 0.1880]    # CNN-M:R=1/6
]



def main(flag):
    linewidth = 1.5
    # ms=6
    fontsize = 10
    # rate_base---psnr
    R = [0.125, 0.25, 0.5, 0.75, 1, 1.5]
    #    1/24   1/12  1/6  1/4  1/3 1/2
    r = [i/3 for i in R]
    L = [25, 50, 75, 100, 125, 150, 175, 200]
    markersize = 8
    if flag == 0:
        plt.figure(figsize=(7, 5))
        plt.subplot(1, 2, 1)
        plt.plot(L, r0125_L_psnr, 'k', marker='o', markerfacecolor='none', markersize=markersize, label="R=1/24")
        plt.plot(L, r05_L_psnr, 'b', marker='s', markerfacecolor='none', markersize=markersize, label="R=1/6")

        max_indx = np.argmax(r0125_L_psnr)  # max value index
        min_indx = np.argmin(r0125_L_psnr)  # min value index
        plt.plot(L[max_indx], r0125_L_psnr[max_indx], 'ks')
        show_max = str(round(r0125_L_psnr[max_indx], 2))
        plt.annotate(show_max, xytext=(L[max_indx]-7, r0125_L_psnr[max_indx]+0.2), xy=(L[max_indx], r0125_L_psnr[max_indx]))
        plt.plot(L[min_indx], r0125_L_psnr[min_indx], 'ks')
        show_min = str(round(r0125_L_psnr[min_indx], 2))
        plt.annotate(show_min, xytext=(L[min_indx]-7, r0125_L_psnr[min_indx]+0.2), xy=(L[min_indx], r0125_L_psnr[min_indx]))

        max_indx = np.argmax(r05_L_psnr)  # max value index
        min_indx = np.argmin(r05_L_psnr)  # min value index
        plt.plot(L[max_indx], r05_L_psnr[max_indx], 'ks')
        show_max = str(round(r05_L_psnr[max_indx], 2))
        plt.annotate(show_max, xytext=(L[max_indx]-7, r05_L_psnr[max_indx]+0.2), xy=(L[max_indx], r05_L_psnr[max_indx]))
        plt.plot(L[min_indx], r05_L_psnr[min_indx], 'ks')
        show_min = str(round(r05_L_psnr[min_indx], 2))
        plt.annotate(show_min, xytext=(L[min_indx]-7, r05_L_psnr[min_indx]+0.2), xy=(L[min_indx], r05_L_psnr[min_indx]))
        plt.xlabel(r"$\alpha$", fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        plt.ylim((18, 25))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(L, r0125_L_ssim, 'k', marker='o', markerfacecolor='none', markersize=markersize, label="R=1/24")
        plt.plot(L, r05_L_ssim, 'b', marker='s', markerfacecolor='none', markersize=markersize, label="R=1/6")

        max_indx = np.argmax(r0125_L_ssim)  # max value index
        min_indx = np.argmin(r0125_L_ssim)  # min value index
        plt.plot(L[max_indx], r0125_L_ssim[max_indx], 'ks')
        show_max = str(round(r0125_L_ssim[max_indx], 2))
        plt.annotate(show_max, xytext=(L[max_indx] - 7, r0125_L_ssim[max_indx] + 0.01),
                     xy=(L[max_indx], r0125_L_ssim[max_indx]))
        plt.plot(L[min_indx], r0125_L_ssim[min_indx], 'ks')
        show_min = str(round(r0125_L_ssim[min_indx], 2))
        plt.annotate(show_min, xytext=(L[min_indx] - 7, r0125_L_ssim[min_indx] + 0.01),
                     xy=(L[min_indx], r0125_L_ssim[min_indx]))

        max_indx = np.argmax(r05_L_ssim)  # max value index
        min_indx = np.argmin(r05_L_ssim)  # min value index
        plt.plot(L[max_indx], r05_L_ssim[max_indx], 'ks')
        show_max = str(round(r05_L_ssim[max_indx], 2))
        plt.annotate(show_max, xytext=(L[max_indx] - 7, r05_L_ssim[max_indx] + 0.01),
                     xy=(L[max_indx], r05_L_ssim[max_indx]))
        plt.plot(L[min_indx], r05_L_ssim[min_indx], 'ks')
        show_min = str(round(r05_L_ssim[min_indx], 2))
        plt.annotate(show_min, xytext=(L[min_indx] - 7, r05_L_ssim[min_indx] + 0.01),
                     xy=(L[min_indx], r05_L_ssim[min_indx]))
        plt.xlabel(r"$\alpha$", fontsize=fontsize)
        plt.ylabel('SSIM', fontsize=fontsize)
        plt.ylim((0.4, 1))
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    elif flag == 1:
        plt.figure(figsize=(7, 5))
        plt.subplot(1, 2, 1)
        goal = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        goal2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        plt.plot(L, goal, 'r--', label="upper-threshold")
        plt.plot(L, goal2, 'r-.', label="good-threshold")
        plt.plot(L, r0125_L_rll3, 'k', marker='o', markerfacecolor='none', markersize=markersize, label="R=1/24")
        plt.plot(L, r05_L_rll3, 'b', marker='s', markerfacecolor='none', markersize=markersize, label="R=1/6")
        plt.xlabel(r"$\alpha$", fontsize=fontsize)
        plt.ylabel(' Proportion of homolopymer run-length>5(%)', fontsize=fontsize)
        plt.xlim((15, 210))
        plt.legend(loc='upper right')
        plt.ylim((0.5, 2.5))
        plt.legend()

        plt.subplot(1, 2, 2)
        goal = [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
        goal2 = [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
        plt.plot(L, goal, 'r--', label="upper-threshold")
        plt.plot(L, goal2, 'r-.', label="down-threshold")
        plt.plot(L, r0125_L_gc2, 'k', marker='o', markerfacecolor='none', markersize=markersize, label="R=1/24")
        plt.plot(L, r05_L_gc2, 'b', marker='s', markerfacecolor='none', markersize=markersize, label="R=1/6")
        plt.xlabel(r"$\alpha$", fontsize=fontsize)
        plt.ylabel('GC-content(%)', fontsize=fontsize)
        plt.xlim((15, 210))
        plt.ylim((40, 60))
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    elif flag == 2:
        plt.figure(3)
        labels = ['1', '2', '3', '4', '5', '6', '7+']
        x = np.arange(len(labels))
        width = 0.8
        r_noloss = [[89.4432, 30.9885, 12.0839, 5.4212, 2.5165, 1.5066, 9.3077],  #r = 0.125  nomyloss
                    [90.2859, 30.9486, 12.2261, 5.4995, 2.7338, 1.4620, 8.4845]  # r = 0.5  nomyloss
                    ]
        r_noloss, r = cul_per_RLL(r_noloss)

        plt.bar(x - width / 2, r0125_L_rll2[2][:], width / 6, label=r"Proposed-1:R=1/24", color="w", edgecolor="k", hatch="...")
        plt.bar(x - width / 3, r05_L_rll2[2][:],   width / 6,  label=r"Proposed-1:R=1/6", color="w", edgecolor="k", hatch="///")
        plt.bar(x - width / 6, r0125_L_rll2[6][:], width / 6,  label=r"Proposed-2:R=1/24", color="w", edgecolor="k", hatch="\\\\")
        plt.bar(x            , r05_L_rll2[6][:],   width / 6,  label=r"Proposed-2:R=1/6", color="w", edgecolor="k", hatch="+")
        plt.bar(x + width / 6, r_noloss[0][:],  width / 6,  label=r"CNN-M:R=1/24", color="w", edgecolor="k", hatch="x")
        plt.bar(x + width / 3, r_noloss[1][:],    width / 6,  label=r"CNN-M:R=1/6", color="w", edgecolor="k", hatch=".")

        plt.xlabel('homologous run-length', fontsize=fontsize)
        plt.xticks(x, ('1', '2', '3', '4', '5', '6', '7+'))
        plt.ylabel('Proportion of homolopymer run-length(%)', fontsize=fontsize)
        plt.ylim((0, 80))
        # plt.grid()
        plt.legend()

    elif flag == 3:
        plt.figure(4)
        labels = ['6', '7+']
        x = np.arange(len(labels))
        width = 0.8
        r_noloss = [[89.4432, 30.9885, 12.0839, 5.4212, 2.5165, 1.5066, 9.3077],  #r = 0.125  nomyloss
                    [90.2859, 30.9486, 12.2261, 5.4995, 2.7338, 1.4620, 8.4845]  # r = 0.5  nomyloss
                    ]
        r_noloss, r = cul_per_RLL(r_noloss)

        plt.bar(x - width / 2, r0125_L_rll2[2][5:7], width / 6,  color="w", edgecolor="k", hatch="...")
        plt.bar(x - width / 3, r05_L_rll2[2][5:7],   width / 6,  color="w", edgecolor="k", hatch="///")
        plt.bar(x - width / 6, r0125_L_rll2[6][5:7], width / 6,   color="w", edgecolor="k", hatch="\\\\")
        plt.bar(x            , r05_L_rll2[6][5:7],   width / 6,  color="w", edgecolor="k", hatch="+")
        plt.bar(x + width / 6, r_noloss[0][5:7],  width / 6,   color="w", edgecolor="k", hatch="x")
        plt.bar(x + width / 3, r_noloss[1][5:7],    width / 6,  color="w", edgecolor="k", hatch=".")
        goal = [1.0, 1.0, 1.0, 1.0]
        plt.plot([-1, 0, 1, 2], goal, 'r--', label="upper-threshold")
        #plt.xlabel('Length of homologous run', fontsize=fontsize)
        plt.xticks(x, ( '6', '7+'))
        #plt.ylabel('Proportion(%)', fontsize=fontsize)
        plt.ylim((0, 10))
        plt.xlim((-0.5, 1.5))
        plt.legend(loc=2)

    elif flag == 4:
        plt.figure(5)

        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        # C1_agct = [i*100 for i in C1_agct]
        x = np.arange(len(labels))
        GC = [[0.2294921875, 0.24609375, 0.25927734375, 0.26513671875],  # Proposed-1:R=1/24
              [0.24609375, 0.26171875, 0.2554931640625, 0.2366943359375],  # Proposed-1:R=1/6
              [0.228515625, 0.27587890625, 0.2548828125, 0.24072265625],  # Proposed-2:R=1/24
              [0.23876953125, 0.255126953125, 0.2681884765625, 0.2379150390625],  # Proposed-2:R=1/6
              [0.1622, 0.3014, 0.2952, 0.2412],  # CNN-M:R=1/24
              [0.2417, 0.2753, 0.2950, 0.1880]  # CNN-M:R=1/6
              ]
        y = torch.tensor(GC) * 100
        width = 0.8  # the width of the bars
        plt.bar(x - width / 2, y[:, 0], width / 4, label='A', color="w", edgecolor="k", linewidth='0.5', hatch="\\\\")
        plt.bar(x - width / 4, y[:, 1], width / 4,  label='C', color="w", edgecolor="k", linewidth='0.5', hatch="//")
        plt.bar(x, y[:, 2], width / 4, label='G', color="w", edgecolor="k", linewidth='0.5', hatch="...")
        plt.bar(x + width / 4, y[:, 3], width / 4, label='T', color="w", edgecolor="k", linewidth='0.5', hatch=".")
        plt.xlabel('Method', fontsize=fontsize)
        plt.xticks(x, ('a', 'b', 'c', 'd', 'e', 'f'))
        plt.ylabel('Proportion(%)', fontsize=fontsize)
        plt.ylim((0, 50))
        plt.legend(loc=2)
        ax2 = plt.twinx()
        ax2.set_ylabel("Proportion of homolopymer run-length(%)", fontsize=fontsize)
        ax2.set_ylim([30, 70])
        y2 = cul_gc(GC)
        plt.plot(x, y2, color='b', marker='o', ms=5, linewidth='1', label="GC-content")
        # 显示数字
        for a, b in zip(x, y2):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=fontsize)
        # 在右侧显示图例
        goal = [55.0, 55.0, 55.0, 55.0, 55.0, 55.0]
        goal2 = [45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
        plt.plot(x, goal, 'r--', label="upper-threshold")
        plt.plot(x, goal2, 'r-.', label="down-threshold")
        plt.legend(loc=1)

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    elif flag == 5:
        R = [0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
        #    1/24   1/12  1/6  1/4  1/3  5/12  1/2
        r = [i / 3 for i in R]
        z_dim_VAEU = [6,     8,     12,   24,   32,    40,   64]
        #            1/64  3/128  3/64   3/32  1/8    5/32   1/4
        r3 = [i * 4 / 32 / 32 for i in z_dim_VAEU]
        Psnr_VAEN = [12.7097, 12.6241, 12.5986, 12.3855, 12.2580, 12.1802, 11.9694]
        Psnr_VAEN2 = [15.7176, 16.2598, 17.2590, 17.2590, 17.5644, 17.5905, 17.6505]

        ssim_VAEN = [0.1875, 0.2025, 0.2187, 0.2369, 0.2300, 0.2224, 0.2193]  # 有信道
        ssim_VAEN2 = [0.2465, 0.3193,  0.3193,  0.3688, 0.3840, 0.3883, 0.3898]  # 无信道

        #        plt.plot(r, C2L1_psnr[4][:], linewidth=linewidth, linestyle='-', marker='^', ms=ms,
        #                 label="Proposed $\gamma_{train}=\gamma_{test}=1.00\%$")
        #        plt.plot(r, C2L1_psnr[0][:], linewidth=linewidth, linestyle='-', marker='^', ms=ms,
        #                 label="Proposed $\gamma_{train}=1.00\%$, $\gamma_{test}=0.00\%$")

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)

        plt.semilogx(r, C1_psnr[0][:], 'b', linewidth=linewidth, linestyle='-', marker='o', markerfacecolor='none', markersize=markersize,
                     label="Proposed-1: $\gamma_{test}=0.50\%$")
        plt.semilogx(r, C1_psnr[1][:], 'k', linewidth=linewidth, linestyle='-.', marker='o', markerfacecolor='none', markersize=markersize,
                     label="Proposed-1: $\gamma_{test}=0.00\%$")

        plt.semilogx(r, C1_psnr[2][:], 'b', linewidth=linewidth, linestyle='-', marker='s', markerfacecolor='none', markersize=markersize,
                     label="Proposed-2: $\gamma_{test}=0.50\%$")
        plt.semilogx(r, C1_psnr[3][:], 'k', linewidth=linewidth, linestyle='-.', marker='s', markerfacecolor='none', markersize=markersize,
                     label="Proposed-2: $\gamma_{test}=0.00\%$")
        plt.semilogx(r3, Psnr_VAEN, 'b', linewidth=linewidth, linestyle='-', marker='d', markerfacecolor='none', markersize=markersize,
                     label="VAEU-QC: $\gamma_{test}=0.50\%$")
        plt.semilogx(r3, Psnr_VAEN2, 'k', linewidth=linewidth, linestyle='-.', marker='d', markerfacecolor='none', markersize=markersize,
                     label="VAEU-QC: $\gamma_{test}=0.00\%$")




        plt.xlabel('R(nts/pixel)', fontsize=fontsize)
        plt.ylabel('PSNR(dB)', fontsize=fontsize)
        #plt.grid()
        plt.ylim((10, 30))
        plt.legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=3, borderaxespad=0)
        # plt.legend(loc=4)



        plt.subplot(1, 2, 2)

        plt.semilogx(r, C1_ssim[0][:], 'b', linewidth=linewidth, linestyle='-', marker='o', markerfacecolor='none', markersize=markersize,
                     label="Proposed-1: $\gamma_{test}=0.50\%$")
        plt.semilogx(r, C1_ssim[1][:], 'k', linewidth=linewidth, linestyle='-.', marker='o', markerfacecolor='none', markersize=markersize,
                     label="Proposed-1: $\gamma_{test}=0.00\%$")

        plt.semilogx(r, C1_ssim[2][:], 'b', linewidth=linewidth, linestyle='-', marker='s', markerfacecolor='none', markersize=markersize,
                     label="Proposed-2: $\gamma_{test}=0.50\%$")
        plt.semilogx(r, C1_ssim[3][:], 'k', linewidth=linewidth, linestyle='-.', marker='s', markerfacecolor='none', markersize=markersize,
                     label="Proposed-2: $\gamma_{test}=0.00\%$")
        plt.semilogx(r3, ssim_VAEN, 'b', linewidth=linewidth, linestyle='-', marker='d', markerfacecolor='none', markersize=markersize,
                     label="VAEU-QC: $\gamma_{test}=0.50\%$")
        plt.semilogx(r3, ssim_VAEN2, 'k', linewidth=linewidth, linestyle='-.', marker='d', markerfacecolor='none', markersize=markersize,
                     label="VAEU-QC: $\gamma_{test}=0.00\%$")


        plt.xlabel('R(nts/pixel)', fontsize=fontsize)
        plt.ylabel('SSIM', fontsize=fontsize)
        #plt.grid()
        plt.ylim((0, 1))
        #plt.legend(ncol = 2)
        # plt.legend(loc="lower left", ncol=len(df.columns))
        #plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0)  # bbox_to_anchor = (1.05, 1), loc=2, borderaxespad=0

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)



    elif flag == 6:
        plt.figure(7)
        test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #  train_error = 0.005
        p_test_0125_p1_train05 = [20.0047, 19.7744, 19.6432, 19.4215, 19.2349, 18.9562, 18.7193, 18.6813, 18.4788, 18.2969]  #75
        p_test_0125_p2_train05 = [19.6727, 19.4602, 19.3442, 19.1498, 18.9768, 18.6980, 18.4461, 18.4222, 18.2631, 18.0822]  #175
        s_test_0125_p1_train05 = [0.6073, 0.5964, 0.5885, 0.5749, 0.5642, 0.5500, 0.5402, 0.5371, 0.5280, 0.5188]  # 75
        s_test_0125_p2_train05 = [0.5918, 0.5804, 0.5733, 0.5610, 0.5498, 0.5366, 0.5270, 0.5232, 0.5155, 0.5072]  # 175

        #  train_error = 0.075
        p_test_0125_p1_train075 = [19.5805, 19.4027, 19.3006, 19.1180, 18.9786, 18.7564, 18.5855, 18.5639, 18.3764, 18.2221]  # 75
        p_test_0125_p2_train075 = [19.4141, 19.2558, 19.1740, 19.0482, 18.9215, 18.7096, 18.5032, 18.5076, 18.3668, 18.2241]   # 175
        s_test_0125_p1_train075 = [0.5943, 0.5841, 0.5775, 0.5638, 0.5532, 0.5421, 0.5336, 0.5298, 0.5207, 0.5121]  # 75
        s_test_0125_p2_train075 = [0.5716, 0.5627, 0.5569, 0.5475, 0.5387, 0.5273, 0.5181, 0.5160, 0.5093, 0.5008]  # 175

        plt.subplot(1,2,1)
        plt.plot(test, p_test_0125_p2_train05, 'k', marker='o', markerfacecolor='none', markersize=markersize, label=r"Proposed-2:$\gamma_{train}=0.5%$" )
        plt.plot(test, p_test_0125_p2_train075, 'k--', marker='s', markerfacecolor='none', markersize=markersize, label=r"Proposed-2:$\gamma_{train}=0.75%$")
        plt.plot(test, p_test_0125_p1_train05, 'b', marker='o', markerfacecolor='none', markersize=markersize, label=r"Proposed-1:$\gamma_{train}=0.5%$")
        plt.plot(test, p_test_0125_p1_train075, 'b--', marker='s', markerfacecolor='none', markersize=markersize, label=r"Proposed-1:$\gamma_{train}=0.75%$")
        plt.xlabel(r"$\gamma_{test}$", fontsize=fontsize)
        plt.ylabel('PSNR', fontsize=fontsize)
        #plt.grid()
        plt.legend()
        plt.ylim((18, 21))

        plt.subplot(1,2,2)
        plt.plot(test, s_test_0125_p2_train05, 'k', marker='o', markerfacecolor='none', markersize=markersize, label=r"Proposed-2:$\gamma_{train}=0.5%$")
        plt.plot(test, s_test_0125_p2_train075, 'k--', marker='s', markerfacecolor='none', markersize=markersize, label=r"Proposed-2:$\gamma_{train}=0.75%$")
        plt.plot(test, s_test_0125_p1_train05, 'b', marker='o', markerfacecolor='none', markersize=markersize, label=r"Proposed-1:$\gamma_{train}=0.5%$")
        plt.plot(test, s_test_0125_p1_train075, 'b--', marker='s', markerfacecolor='none', markersize=markersize, label=r"Proposed-1:$\gamma_{train}=0.75%$")

        plt.xlabel(r"$\gamma_{test}$", fontsize=fontsize)
        plt.ylabel('SSIM', fontsize=fontsize)
        # plt.grid()
        plt.ylim((0.5, 0.7))
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)


    elif flag == 7:

        plt.figure(8)
        test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        p1_r0125_psnr = [20.0047, 19.7744, 19.6432, 19.4215, 19.2349, 18.9562, 18.7193, 18.6813, 18.4788, 18.2969]
        p1_r0125_ssin = [0.6073, 0.5964, 0.5885, 0.5749, 0.5642, 0.5500, 0.5402, 0.5371, 0.5280, 0.5188]
        p1_r0125_psnr_m = [20.1379, 19.9516, 19.8691, 19.5925, 19.4119, 19.2887, 19.0873, 18.9742, 18.9058, 18.6766]
        p1_r0125_ssin_m = [0.6266, 0.6153, 0.6092, 0.5922, 0.5804, 0.5731, 0.5626, 0.5547, 0.5514, 0.5406]

        p2_r0125_psnr = [19.6727, 19.4602, 19.3442, 19.1498, 18.9768, 18.6980, 18.4461, 18.4222, 18.2631, 18.0822]
        p2_r0125_ssin = [0.5918, 0.5804, 0.5733, 0.5610, 0.5498, 0.5366, 0.5270, 0.5232, 0.5155, 0.5072]
        p2_r0125_psnr_m = [19.7762, 19.6239, 19.5592, 19.3579, 19.2146, 19.0848, 18.8815, 18.7792, 18.6948, 18.4862]
        p2_r0125_ssin_m = [0.5989, 0.5906, 0.5863, 0.5743, 0.56526, 0.5582, 0.5485, 0.5425, 0.5393, 0.5301]

        plt.subplot(1, 2, 1)
        plt.plot(test, p1_r0125_psnr, 'k', marker='o', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-1:$v=2$")
        plt.plot(test, p1_r0125_psnr_m, 'k--', marker='s', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-1:$v=4$")
        plt.plot(test, p2_r0125_psnr, 'b', marker='o', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-2:$v=2$")
        plt.plot(test, p2_r0125_psnr_m, 'b--', marker='s', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-2:$v=4$")
        plt.xlabel(r"$\gamma_{test}$", fontsize=fontsize)
        plt.ylabel('PSNR', fontsize=fontsize)
        # plt.grid()
        plt.legend()
        plt.ylim((18, 21))

        plt.subplot(1, 2, 2)
        plt.plot(test, p1_r0125_ssin, 'k', marker='o', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-1:$v=2$")
        plt.plot(test, p1_r0125_ssin_m, 'k--', marker='s', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-1:$v=4$")
        plt.plot(test, p2_r0125_ssin, 'b', marker='o', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-2:$v=2$")
        plt.plot(test, p2_r0125_ssin_m, 'b--', marker='s', markerfacecolor='none', markersize=markersize,
                 label=r"Proposed-2:$v=4$")

        plt.xlabel(r"$\gamma_{test}$", fontsize=fontsize)
        plt.ylabel('SSIM', fontsize=fontsize)
        # plt.grid()
        plt.ylim((0.5, 0.7))
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)




if __name__ == '__main__':

    for i in range(5,6):
        main(i)
    plt.show()

