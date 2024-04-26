import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(MConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

# class ConvEncoder(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(ConvEncoder, self).__init__()
#
#         self.conv_block = nn.Sequential(
#             nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(output_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(output_channels, output_channels, kernel_size=1),
#             nn.BatchNorm1d(output_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(output_channels, output_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#         x = self.conv_block(x.permute(0, 2, 1))
#         return x

# # 使用示例
# input_channels = 5  # 输入数据的通道数
# output_channels = 32  # 输出数据的通道数
#
#
# encoder = ConvEncoder(input_channels, output_channels)
# input_data = torch.randn(32, input_channels, 50)  # 生成随机输入数据
# output = encoder(input_data)
# print("Encoder output shape:", output.shape)


class ConvFFN(nn.Module):
    def __init__(self, D, r=1, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        self.pw_con = nn.Conv1d(
            in_channels=D,
            out_channels=r*D,
            kernel_size=1,
            groups=D
            )
    def forward(self, x):
        # x: [B, D, N]
        x = self.pw_con(F.gelu(self.pw_con(x)))
        return x  # x: [B, D, N]

class ConvEncoder(nn.Module):
    def __init__(self, D, kernel_size=51, r=1):
        super(ConvEncoder, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            padding=(kernel_size-1)//2
            # padding='same'
            )
        self.bn = nn.BatchNorm1d(D)
        self.conv_ffn = ConvFFN(D, r, one=True)


    def forward(self, x_emb):
        # x_emb: [B, M, D, N]/ [B, length // scale, scale, N]
        B, M, D, N = x_emb.shape
        x = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x.permute(0, 2, 1))                                   # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)                                       # [B, N, M*D]
        x = self.conv_ffn(x)                                 # [B, N, M*D]
        x = x.reshape(B, N, M, D).permute(0,2,3,1)           # [B, N, M, D] -> [B, M, D, N]
        out = x + x_emb
        return out  # out: [B, M, D, N]