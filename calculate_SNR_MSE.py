'''
calculate the SNR and MSE
'''
import os
import math
import numpy as np
import glob
from natsort import natsorted
from scipy.io.wavfile import read
import librosa


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = './path'
    folder_Gen = './path'
    # folder_Gen = '/media/l228/数据/zxh/HSA_updwnsp_dwt/audios_0.1/secret-rev/'
    # crop_border = 1
    # suffix = '_secret_rev'  # suffix for Gen images
    test_norm = False  # True: 归一化; False:不归一化

    SNR_all = []
    MSE_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    if test_norm:
        print('Testing norm.')
    else:
        print('Testing turth value.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        _, au_GT = read(img_path)
        _, au_Gen = read(os.path.join(folder_Gen, base_name + '.wav'))


        im_GT_in = au_GT
        im_Gen_in = au_Gen


        # calculate SNR and MSE
        SNR = computeSNR(im_GT_in, im_Gen_in)
        MSE = computeMSE(im_GT_in, im_Gen_in)

        print('{:3d} - {:25}. \tSNR: {:.6f} dB, \tMSE: {:.10f}'.format(
            i + 1, base_name, SNR, MSE))
        SNR_all.append(SNR)
        MSE_all.append(MSE)
    print('Average: SNR: {:.6f} dB, MSE: {:.10f}'.format(
        sum(SNR_all) / len(SNR_all),
        sum(MSE_all) / len(MSE_all)))


def computeSNR(origin, pred):
    origin = origin.astype(np.float32)  # 转成32位 8*3=24
    pred = pred.astype(np.float32)
    p_signal = np.sum(origin ** 2)
    p_noise = np.sum((origin / 1.0 - pred / 1.0) ** 2)  # 求mse
    print(p_signal)
    print(p_noise)
    if p_signal == 0:
        p_signal = 1

    return 10 * math.log10(p_signal / p_noise)

def computeMSE(origin, pred):
    origin = origin.astype(np.float32)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/32768.0 - pred/32768.0)**2)
    # mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    return mse




if __name__ == '__main__':
    main()
