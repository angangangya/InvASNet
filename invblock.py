from math import exp
import torch
import torch.nn as nn
from up_down_sp import Up_Down_Net
# from CRNN import Bottle_RNN_FC
# from transformer import Transformer
import config as c


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Up_Down_Net, clamp=c.clamp, harr=False, in_1=c.channels_in, in_2=c.channels_in):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 2  # 小波变换后，通道变成原来的4倍
            self.split_len2 = in_2 * 2
        else:
            self.split_len1 = in_1
            self.split_len2 = in_2
        self.clamp = clamp

        # 这几个函数都是dense
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)
        # self.r = subnet_constructor
        # self.y = subnet_constructor
        # self.f = subnet_constructor
        # self.p = subnet_constructor

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        # narrow变换变量某一维度的某几个值，取变量x的第1维中从【a，b】的值，此处取通道数，载体图像和imp属于x1，秘密图像属于x2
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:  # 正向过程

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped! 逆向过程

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)

        return torch.cat((y1, y2), 1)  # 在第1维度上进行拼接，也就是通道C上进行拼接


# test
def main():
    x = torch.randn(16, 4, 4096)
    net = INV_block_affine()
    y = net(x, True)
    print(y.shape)


if __name__ == '__main__':
    main()
