import torch
import torch.nn as nn


class TripletCoordAtt(nn.Module):
    def __init__(self, k_size=3):
        super(TripletCoordAtt, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_c = nn.AdaptiveAvgPool2d(1)

        self.conv_h = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)
        self.conv_w = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)
        self.conv_c = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_w = self.pool_w(x).transpose(-1, -2).squeeze(-1)
        x_h = self.pool_h(x).squeeze(-1)
        x_c = self.pool_c(x).squeeze(-1).transpose(-1, -2)

        o_w = self.conv_w(x_w).unsqueeze(-1).transpose(-1, -2)
        o_h = self.conv_w(x_h).unsqueeze(-1)
        o_c = self.conv_w(x_c).transpose(-1, -2).unsqueeze(-1)

        out = identity * o_w * o_h * o_c

        return out
def main():
    attention_block = TripletCoordAtt()
    input = torch.rand([4,64,32,32])
    output = attention_block(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    main()