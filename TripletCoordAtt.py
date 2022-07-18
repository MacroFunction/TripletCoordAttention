import torch
import torch.nn as nn
from torchsummary import summary


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

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_w = self.pool_w(x).transpose(-1, -2).squeeze(-1)
        x_h = self.pool_h(x).squeeze(-1)
        x_c = self.pool_c(x).squeeze(-1).transpose(-1, -2)

        o_h = self.sigmoid(self.conv_h(x_h).unsqueeze(-1))
        o_w = self.sigmoid(self.conv_w(x_w).unsqueeze(-1).transpose(-1, -2))
        o_c = self.sigmoid(self.conv_c(x_c).transpose(-1, -2).unsqueeze(-1))

        out = identity * o_w * o_h * o_c
        return out


def main():
    attention_block = TripletCoordAtt()
    input = torch.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attention_block = TripletCoordAtt().to(device)
    input = torch.rand([4,64,32,32])
    input = input.to(device)
    output = attention_block(input)
    print(attention_block)
    # # model.load_state_dict(torch.load('./models/model75.78.pth'), strict=False)
    #
    # # x = torch.rand(size=(3, 224, 224))
    summary(attention_block, (3, 224, 224))
    # print(input.size(), output.size())

if __name__ == '__main__':
    main()