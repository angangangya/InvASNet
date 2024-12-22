#!/usr/bin/env python
import sys
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
from audio_datastets import trainloader, testloader, valloader
import viz
import modules.Unet_common as common
import warnings
from Noise import Noise
warnings.filterwarnings("ignore")

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)  # 其实就是不带换行的print
        self.log.write(message)  # 写到文件里
        self.log.flush()  # 刷新输出

    def flush(self):
        pass


def computeSNR(origin, pred):
    origin = np.array(origin)  # 转成数组
    origin = origin.astype(np.float32)  # 转成32位 8*3=24
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    p_signal = np.sum(origin ** 2)
    p_noise = np.sum((origin / 1.0 - pred / 1.0) ** 2)  # 求mse

    return 10 * math.log10(p_signal / p_noise)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda() # 为每个batch建立一个[c,h,w]的高斯分布的随机数，均值为0，方差为1,noise[0]的shape是[c,h,w]

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss.cuda()


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(rev_input, input)
    return loss.cuda()


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.cuda()


def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.cuda()


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# def load(name, net):
#     state_dicts = torch.load(name)
#     network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
#     net.load_state_dict(network_state_dict)
#     try:
#         optim.load_state_dict(state_dicts['opt'])  # 优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)
#     except:
#         print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
net = Model()  # 隐藏第一张图的网络
init_model(net)

# 用多个GPU来加速训练
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
net.cuda()

# 计算参数数量
para = get_parameter_number(net)
print(para)

# 找到可以训练的参数
# filter用于过滤序列，过于掉不符合条件的元素，返回由符合条件元素组成的新列表，参数中第一个是函数，用来判断，第二个是序列
# lambda创建匿名函数，函数表示：需要保留梯度
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

# DWT
dwt = common.DWT1d()
iwt = common.IWT1d()

quantization = Noise()

# # 中断后接着训练
# if c.train_next:
#     load(c.MODEL_PATH + c.suffix_load + '.pt', net)
#
# # 加载之前训练的模型
# if c.pretrain:
#     load(c.PRETRAIN_PATH + c.suffix_pretrain + '.pt', net)
#
# # 开始新的训练
# if c.new_train:
#     c.trained_epoch = 0

# 优化器 params_trainable1是待优化的参数
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

# 使用学习率衰减
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])  # 优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)
    except:
        print('Cannot load optimizer for some reason or other')


# 中断后接着训练
if c.train_next:
    load(c.MODEL_PATH + c.suffix_load + '.pt', net)

# 加载之前训练的模型
if c.pretrain:
    load(c.PRETRAIN_PATH + c.suffix_pretrain + '.pt', net)

try:
    # 可视化，生成的日志存到默认文件夹runs下，日志文件的后缀为steg，runs下的文件名的后缀为hinet
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")
    log = Logger('3_30.log', sys.stdout)

    for i_epoch in range(c.epochs):
        net.train()
        i_epoch = i_epoch + c.trained_epoch + 1  # 以前训练过的次数是trained_epoch
        loss_history = []
        loss_history_g = []
        loss_history_r = []
        loss_history_l = []

        #################
        #     train:    #
        #################
        for i_batch, data in enumerate(trainloader):
            # data preparation
            cover = data['music'].cuda().unsqueeze(1)  # [b, 1, 44160]
            secret = data['speech'].cuda().unsqueeze(1)  # [b, 1, 44160]

            # dwt
            cover_dwt = dwt(cover)  # channel = 2 [b, 2, 22080]
            cover_dwt_low = cover_dwt.narrow(1, 0, 1)  # 低频
            secret_dwt = dwt(secret)  # channel = 2 [b, 2, 22080]

            input_data = torch.cat((cover_dwt, secret_dwt), 1)  # channels = 4

            #################
            #    forward1:   #
            #################
            output_data_dwt = net(input_data)  # channels = 4
            output_steg_dwt = output_data_dwt.narrow(1, 0, c.channels_in)  # channels = 2 载密
            # output_steg_dwt_low = output_steg_dwt.narrow(1, 0, 1)
            output_z = output_data_dwt.narrow(1, c.channels_in, c.channels_in)  # channels = 2 残差

            # get steg
            output_steg = iwt(output_steg_dwt)

            # quantization
            quant_steg = quantization(output_steg, train=True)
            # quant_steg = torch.clamp(torch.round(output_steg * 32768.), -32768, 32767)/32768 - output_steg.detach() + output_steg

            #################
            #   backward1:   #
            #################
            output_z_guass = gauss_noise(output_z.shape)  # channels = 1
            quant_steg_dwt = dwt(quant_steg)
            quant_steg_dwt_low = quant_steg_dwt.narrow(1, 0, 1)  # low
            output_rev = torch.cat((quant_steg_dwt, output_z_guass), 1)  # channels = 2

            rev_data_dwt = net(output_rev, rev=True)  # channels = 2
            rev_secret_dwt = rev_data_dwt.narrow(1, c.channels_in, c.channels_in)  # channels = 1 恢复的第一张秘密图像频域

            # get secret_rev
            rev_secret = iwt(rev_secret_dwt)
            quant_rev_secret = quantization(rev_secret, train=True)
            #################
            #     loss:     #
            #################
            g_loss = guide_loss(quant_steg.cuda()*32768, cover.cuda()*32768)  # 载体图像和第一幅载密图像的loss
            r_loss = reconstruction_loss(quant_rev_secret.cuda()*32768, secret.cuda()*32768)  # 秘密图像和第一幅恢复的秘密图像的loss L1loss
            l_loss = guide_loss(quant_steg_dwt_low.cuda()*32768, cover_dwt_low.cuda()*32768)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()  # 求导，反向传播计算得到每个参数的梯度值

            torch.nn.utils.clip_grad_norm(net.parameters(), 10)
            optim.step()
            # 梯度清零，即上一次的梯度记录被清空
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            loss_history_g.append(g_loss.item())
            loss_history_r.append(r_loss.item())
            loss_history_l.append(l_loss.item())

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():  # 不求梯度
                psnr_s = []
                psnr_c = []
                net.eval()  # 不改变权值
                for data in valloader:
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
                    # quantization
                    quant_steg = quantization(output_steg, train=False)
                    # quant_steg = torch.clamp(torch.round(output_steg * 32768.), -32768, 32767) / 32768

                    #################
                    #   backward1:   #
                    #################
                    output_z_guass = gauss_noise(output_z.shape)  # channels = 1
                    quant_steg_dwt = dwt(quant_steg)
                    output_rev = torch.cat((quant_steg_dwt, output_z_guass), 1)  # channels = 2

                    rev_data_dwt = net(output_rev, rev=True)  # channels = 2
                    rev_secret_dwt = rev_data_dwt.narrow(1, c.channels_in, c.channels_in)  # channels = 1

                    # get secret_rev
                    rev_secret = iwt(rev_secret_dwt)
                    quant_rev_secret = quantization(rev_secret, train=False)

                    # [-1,1]
                    cover_audio = cover.squeeze().cpu().numpy() * c.MAX_WAV_VALUE
                    steg_audio = quant_steg.squeeze().cpu().numpy() * c.MAX_WAV_VALUE

                    secret_audio = secret.squeeze().cpu().numpy() * c.MAX_WAV_VALUE
                    rev_secret_audio = quant_rev_secret.squeeze().cpu().numpy() * c.MAX_WAV_VALUE

                    # [0, 1]
                    # cover_audio = cover.squeeze().cpu().numpy() * 65535 - c.MAX_WAV_VALUE
                    # steg_audio = output_steg.squeeze().cpu().numpy() * 65535 - c.MAX_WAV_VALUE
                    #
                    # secret_audio = secret.squeeze().cpu().numpy() * 65535 - c.MAX_WAV_VALUE
                    # rev_secret_audio = rev_secret.squeeze().cpu().numpy() * 65535 - c.MAX_WAV_VALUE

                    snr_temp_s = computeSNR(secret_audio, rev_secret_audio)  # 计算SNR
                    psnr_s.append(snr_temp_s)

                    snr_temp_c = computeSNR(cover_audio, steg_audio)
                    psnr_c.append(snr_temp_c)

                writer.add_scalars("SNR", {"S average snr": np.mean(psnr_s)}, i_epoch)  # 添加到runs下的summary里
                writer.add_scalars("SNR", {"C average snr": np.mean(psnr_c)}, i_epoch)
                log.write('----------------------------------------------------------------------------------\n')
                log.write('C_SNR: {:.4f}           S_SNR: {:.4f}\n'.format(np.mean(psnr_c), np.mean(psnr_s)))

        # 计算一个epoch的loss值
        epoch_losses = np.mean(np.array(loss_history), axis=0)  # 因为mse函数只是求和，这里再求一次平均
        epoch_losses[1] = optim.param_groups[0]['lr']

        epoch_losses_g = np.mean(np.array(loss_history_g))
        epoch_losses_r = np.mean(np.array(loss_history_r))
        epoch_losses_l = np.mean(np.array(loss_history_l))

        log.write('----------------------------------------------------------------------------------\n')
        log.write('Epoch: {}                   Lr: {:.1E}\n'.format(i_epoch, epoch_losses[1]))
        log.write('Train_Loss: {:.4E}    g_loss: {:.4E}   r_loss: {:.4E}   l_loss:{:.4E}    \n'.format(epoch_losses[0],
                                                                                                        epoch_losses_g,
                                                                                                        epoch_losses_r,
                                                                                                      epoch_losses_l))
        # log.write('C_SNR: {:.4E}           S_SNR: {:.4E}\n'.format(np.mean(psnr_c), np.mean(psnr_s)))
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)  # Train_loss是变量名，后面是要存的值，最后是x坐标
        writer.add_scalars("Train", {"g_Loss": epoch_losses_g}, i_epoch)
        writer.add_scalars("Train", {"r_Loss": epoch_losses_r}, i_epoch)

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')

        # 学习率衰减，一个epoch衰减一次
        weight_scheduler.step()

    # 训练结束保存模型
    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
    writer.close()

except:  # try发生错误之后，执行这个代码
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')

    raise

finally:  # 最后都会执行这个代码
    viz.signal_stop()
