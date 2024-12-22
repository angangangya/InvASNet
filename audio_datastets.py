import torch
import random
from scipy.io.wavfile import read
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional
import config as c
import librosa
import numpy as np


# 获得所有的音频文件名，转成list
def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]  # 去掉换行符
    return files


# def load_wav_to_torch(full_path, mono, speech=False):
#     """
#     Loads wavdata into torch array
#     """
#     # _, data = read(full_path)  # 打开.wav文件，返回采样率int和音频数据numpy  data[n, 2]
#     data, _ = librosa.load(full_path, sr=None, mono=mono)  # 对音频重采样，归一化了  data[2, n]
#     # 嵌入率降低
#     i = 0
#     if speech:
#         # while data[i] == 0:
#         #     i += 1
#         start = random.randint(0, 44160-276)
#         part_data = data[start:start+276]
#         # new_data = np.pad(part_data, (0, 44160 - 4416))
#         new_data = np.pad(part_data, (21942, 21942))
#     else:
#         new_data = data
#     return torch.from_numpy(new_data).float()


def load_wav_to_torch(full_path, mono, speech=False):
    """
    Loads wavdata into torch array
    """
    # _, data = read(full_path)  # 打开.wav文件，返回采样率int和音频数据numpy  data[n, 2]
    data, _ = librosa.load(full_path, sr=None, mono=mono)  # 对音频重采样，归一化了  data[2, n]
    return torch.from_numpy(data).float()


class Audio_Dataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            self.audio_files = files_to_list(c.music_training_files)  # 训练数据转成list
            self.speech_files = files_to_list(c.speech_training_files)
        if self.mode == 'val':
            self.audio_files = files_to_list(c.music_val_files)  # 测试数据转成list
            self.speech_files = files_to_list(c.speech_val_files)
        if self.mode == 'test':
            self.audio_files = files_to_list(c.music_testing_files)  # 测试数据转成list
            self.speech_files = files_to_list(c.speech_testing_files)

    def __getitem__(self, index):
        audio_filename = self.audio_files[index]  # 读取一个music数据
        audio = load_wav_to_torch(audio_filename, mono=True)  # 读取该音频

        speech_filename = self.speech_files[index]  # 读取一个speech数据
        speech = load_wav_to_torch(speech_filename, mono=True, speech=True)  # 读取该音频

        data = {'music': audio, 'speech': speech}
        return data

    def __len__(self):
        return len(self.audio_files)  # 音频数据集数量


# Training data loader
trainloader = DataLoader(
    Audio_Dataset(mode="train"),
    batch_size=c.batch_size,  # 训练时的batchsize是24
    shuffle=True,  # 打乱数据
    pin_memory=False,
    num_workers=0,
    drop_last=True
)
# val data loader
valloader = DataLoader(
    Audio_Dataset(mode="val"),  # 加载的数据集对象
    batch_size=c.batchsize_val,  # 验证时的batchsize是3
    shuffle=False,  # 不打乱数据
    pin_memory=False,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    num_workers=0,  # 使用多进程加载的进程数，0代表不使用多进程
    drop_last=True  # dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
)
# val data loader
testloader = DataLoader(
    Audio_Dataset(mode="test"),  # 加载的数据集对象
    batch_size=c.batchsize_val,  # 验证时的batchsize是3
    shuffle=False,  # 不打乱数据
    pin_memory=False,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    num_workers=0,  # 使用多进程加载的进程数，0代表不使用多进程
    drop_last=True  # dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
)
