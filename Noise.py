import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()

    def forward(selfself, input, train=False):
        input = input * 32768.0
        if train:
            noise = torch.nn.init.uniform_(torch.zeros_like(input), -0.5, 0.5).cuda()
            output = input + noise
            output = torch.clamp(output, -32768, 32767)
        else:
            output = input.round() * 1.0
            output = torch.clamp(output, -32768, 32767)
        return output / 32768.0
