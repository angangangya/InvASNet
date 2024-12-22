import warnings
import sys
import math
import os
import torch
import torch.nn
import torch.optim
import numpy as np
# import cv2
from model import *
import config as c
from os.path import join
import audio_datastets
import modules.module_util as mutil
import modules.Unet_common as common
import time
import librosa
import soundfile
from Noise import Noise


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduction='mean')  # reduce=True返回标量
    loss = loss_fn(output, bicubic_image)
    return loss.cuda()


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(rev_input, input)
    return loss.cuda()


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computeSNR(origin, pred):
    origin = np.array(origin)  # 转成数组
    origin = origin.astype(np.float32)  # 转成32位 8*3=24
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    p_signal = np.sum(origin ** 2)
    p_noise = np.sum((origin / 1.0 - pred / 1.0) ** 2)  # 求mse

    return 10 * math.log10(p_signal / p_noise)


# DWT
dwt = common.DWT1d()
iwt = common.IWT1d()
quantization = Noise()

net = Model()
net.cuda()

init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

load('/media/l228/数据/zxh/HSA_updwnsp_dwt/models-1-7/' + 'model_checkpoint_00050' + '.pt', net)
# load('/media/l228/数据/zxh/HSA_updwnsp_dwt/models/' + 'model_checkpoint_00028' + '.pt', net)

with torch.no_grad():
    net.eval()
    g_loss_total = []
    r_loss_total = []
    psnr_s = []
    psnr_c = []
    start = time.time()
    for i, data in enumerate(audio_datastets.testloader):
        cover = data['music'].cuda().unsqueeze(1)  # channels = 1
        secret = data['speech'].cuda().unsqueeze(1)

        # dwt
        cover_dwt = dwt(cover)  # channel = 2
        secret_dwt = dwt(secret)  # channel = 2

        input_data = torch.cat((cover_dwt, secret_dwt), 1)  # channels = 4

        #################
        #    forward1:   #
        #################
        output_data_dwt = net(input_data)  # channels = 4
        output_steg_dwt = output_data_dwt.narrow(1, 0, c.channels_in)  # channels = 2 载密
        output_z = output_data_dwt.narrow(1, c.channels_in, c.channels_in)  # channels = 2 残差

        # get steg
        output_steg = iwt(output_steg_dwt)

        #################
        #   backward1:   #
        #################
        output_z_guass = gauss_noise(output_z.shape)  # channels = 1
        ### 存残差
        # np.save(c.z_path + '%.5d.npy' % i, output_z.squeeze().cpu().numpy())
        # np.save(c.g_path + '%.5d.npy' % i, output_z_guass.squeeze().cpu().numpy())

        # quantization
        quant_steg = quantization(output_steg, train=False)
#
        quant_steg_dwt = dwt(quant_steg)
        output_rev = torch.cat((quant_steg_dwt, output_z_guass), 1)  # channels = 2

        rev_data_dwt = net(output_rev, rev=True)  # channels = 2
        rev_secret_dwt = rev_data_dwt.narrow(1, c.channels_in, c.channels_in)  # channels = 1

        # get secret_rev
        rev_secret = iwt(rev_secret_dwt)

        cover_audio = cover.squeeze().cpu().numpy()
        steg_audio = quant_steg.squeeze().cpu().numpy()

        secret_audio = secret.squeeze().cpu().numpy()
        rev_secret_audio = rev_secret.squeeze().cpu().numpy()

        # g_loss = guide_loss(output_steg, cover)  # 载体图像和第一幅载密图像的loss
        # r_loss = reconstruction_loss(secret, rev_secret)
        # g_loss_total.append(g_loss.cpu().numpy())
        # r_loss_total.append(r_loss.cpu().numpy())
        #
        # cover_audio = cover.squeeze().cpu().numpy()
        # steg_audio = output_steg.squeeze().cpu().numpy()
        #
        # secret_audio = secret.squeeze().cpu().numpy()
        # rev_secret_audio = rev_secret.squeeze().cpu().numpy()
        #
#         # psnr_temp = computeSNR(secret_audio, rev_secret_audio)  # 计算SNR
#         # psnr_s.append(psnr_temp)
#         #
#         # psnr_temp_c = computeSNR(cover_audio, steg_audio)
#         # psnr_c.append(psnr_temp_c)
#
        soundfile.write(c.TEST_PATH_cover + '%.5d.wav' % i, cover_audio, samplerate=44100, subtype='PCM_16')  # 默认按PCM16保存
        soundfile.write(c.TEST_PATH_steg + '%.5d.wav' % i, steg_audio, samplerate=44100, subtype='PCM_16')
#
#         soundfile.write(c.TEST_PATH_secret + '%.5d.wav' % i, secret_audio[21942:21942+276], samplerate=16000, subtype='PCM_16')
#         soundfile.write(c.TEST_PATH_secret_rev + '%.5d.wav' % i, rev_secret_audio[21942:21942+276], samplerate=16000, subtype='PCM_16')

    end = time.time()
    print('程序执行时间: ', end - start)
    # print(np.mean(psnr_s))
    # print(np.mean(psnr_c))
    # print(np.mean(g_loss_total))
    # print(np.mean(r_loss_total))

