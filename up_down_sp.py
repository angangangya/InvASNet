import torch.nn as nn
import torch


# class Up_Down_Net(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(Up_Down_Net, self).__init__()
#         # down sample
#         self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=15, kernel_size=32, stride=2, padding=15)  # batch,15,8192
#         self.in1 = nn.BatchNorm1d(15)
#         self.conv1_f = nn.PReLU()
#
#         self.conv2 = nn.Conv1d(15, 32, 32, 2, 15)  # batch,32,4096
#         self.in2 = nn.BatchNorm1d(32)
#         self.conv2_f = nn.PReLU()
#
#         self.conv3 = nn.Conv1d(32, 64, 32, 2, 15)  # batch,32,2048
#         self.in3 = nn.BatchNorm1d(64)
#         self.conv3_f = nn.PReLU()
#
#         self.deconv4 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # batch,64,1024
#         self.in4 = nn.BatchNorm1d(32)
#         self.deconv4_f = nn.PReLU()
#
#         self.deconv5 = nn.ConvTranspose1d(64, 15, 32, 2, 15)  # batch,64,256
#         self.in5 = nn.BatchNorm1d(15)
#         self.deconv5_f = nn.PReLU()
#
#         self.deconv6 = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # batch,64,1024
#         self.deconv6_f = nn.PReLU()
#
#     def forward(self, x):
#         c1 = self.conv1(x)
#         # c1 = self.in1(c1)
#         down1 = self.conv1_f(c1)
#
#         c2 = self.conv2(down1)
#         # c2 = self.in2(c2)
#         down2 = self.conv2_f(c2)
#
#         c3 = self.conv3(down2)
#         # c3 = self.in3(c3)
#         down3 = self.conv3_f(c3)
#
#         d4 = self.deconv4(down3)
#         # d4 = self.in4(d4)
#         up4 = self.deconv4_f(torch.cat((d4, down2), 1))
#
#         d5 = self.deconv5(up4)
#         # d5 = self.in5(d5)
#         up5 = self.deconv5_f(torch.cat((d5, down1), 1))
#
#         out = self.deconv6(up5)
#
#         return out

class Up_Down_Net(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Up_Down_Net, self).__init__()
        # down sample
        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=16, kernel_size=32, stride=2, padding=15)  # batch,16,8192
        self.in1 = nn.BatchNorm1d(16)
        self.conv1_f = nn.PReLU()

        self.conv2 = nn.Conv1d(16, 32, 32, 2, 15)  # batch,32,4096
        self.in2 = nn.BatchNorm1d(32)
        self.conv2_f = nn.PReLU()

        self.conv3 = nn.Conv1d(32, 64, 32, 2, 15)  # batch,32,2048
        self.in3 = nn.BatchNorm1d(64)
        self.conv3_f = nn.PReLU()

        self.conv4 = nn.Conv1d(64, 64, 32, 2, 15)  # batch,64,1024
        self.in4 = nn.BatchNorm1d(64)
        self.conv4_f = nn.PReLU()

        self.conv5 = nn.Conv1d(64, 128, 32, 2, 15)  # batch,128,512
        self.in5 = nn.BatchNorm1d(128)
        self.conv5_f = nn.PReLU()

        self.deconv6 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # batch,64,1024
        self.in6 = nn.BatchNorm1d(64)
        self.deconv6_f = nn.PReLU()

        self.deconv7 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # batch,32,2048
        self.in7 = nn.BatchNorm1d(64)
        self.deconv7_f = nn.PReLU()

        self.deconv8 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # batch,32,4096
        self.in8 = nn.BatchNorm1d(32)
        self.deconv8_f = nn.PReLU()

        self.deconv9 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # batch,16,8192
        self.in9 = nn.BatchNorm1d(16)
        self.deconv9_f = nn.PReLU()

        self.deconv10 = nn.ConvTranspose1d(32, output_channel, 32, 2, 15)  # batch,1,1384


    def forward(self, x):
        c1 = self.conv1(x)
        # c1 = self.in1(c1)
        down1 = self.conv1_f(c1)

        c2 = self.conv2(down1)
        # c2 = self.in2(c2)
        down2 = self.conv2_f(c2)

        c3 = self.conv3(down2)
        # c3 = self.in3(c3)
        down3 = self.conv3_f(c3)

        c4 = self.conv4(down3)
        # c4 = self.in4(c4)
        down4 = self.conv4_f(c4)

        c5 = self.conv5(down4)
        # c5 = self.in5(c5)
        down5 = self.conv5_f(c5)

        d6 = self.deconv6(down5)
        # d6 = self.in6(d6)
        down6 = self.deconv6_f(torch.cat((d6, down4), dim=1))

        d7 = self.deconv7(down6)
        # d7 = self.in7(d7)
        down7 = self.deconv7_f(torch.cat((d7, down3), dim=1))

        d8 = self.deconv8(down7)
        # d8 = self.in8(d8)
        down8 = self.deconv8_f(torch.cat((d8, down2), dim=1))

        d9 = self.deconv9(down8)
        # d9 = self.in9(d9)
        up9 = self.deconv9_f(torch.cat((d9, down1), dim=1))

        out = self.deconv10(up9)

        return out


if __name__ == '__main__':
    a = torch.randn(1, 2, 22080)
    net = Up_Down_Net(2, 2)
    b = net(a)
    print(b.shape)
