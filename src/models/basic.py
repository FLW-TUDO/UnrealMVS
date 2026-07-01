import torch
import torch.nn.functional as F


class Conv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1, dilation=1, use_bn=True, relu=True):
        super().__init__()
        self.use_bn = use_bn
        self.relu   = relu
        self.conv   = torch.nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=pad, dilation=dilation)
        if use_bn:
            self.batch_norm = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if residual is not None:
            x += residual
        if self.relu:
            x = F.relu(x)
        return x


class Conv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1, dilation=1, use_bn=True, relu=True):
        super().__init__()
        self.use_bn = use_bn
        self.relu   = relu
        self.conv   = torch.nn.Conv3d(ch_in, ch_out, kernel_size, stride, padding=pad, dilation=dilation)
        if use_bn:
            self.batch_norm = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if residual is not None:
            x += residual
        if self.relu:
            x = F.relu(x)
        return x


class DeConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, out_pad=0, use_bn=True, relu=True):
        super().__init__()
        self.use_bn = use_bn
        self.relu   = relu
        self.conv   = torch.nn.ConvTranspose3d(
            ch_in, ch_out, kernel_size, stride, padding=pad,
            dilation=dilation, output_padding=out_pad,
        )
        if use_bn:
            self.batch_norm = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if residual is not None:
            x += residual
        if self.relu:
            x = F.relu(x)
        return x


class UpsampleConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, out_pad=0, use_bn=True, relu=True):
        super().__init__()
        self.upsample   = torch.nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False)
        self.conv       = torch.nn.Conv3d(ch_in, ch_out, kernel_size, stride=1, padding=pad, dilation=dilation)
        self.use_bn     = use_bn
        self.relu       = relu
        if use_bn:
            self.batch_norm = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.upsample(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if residual is not None:
            x += residual
        if self.relu:
            x = F.relu(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] // 2, x.shape[2] * 2, x.shape[3], x.shape[4])
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class CutTensor(torch.nn.Module):
    def forward(self, x):
        x   = x.view(x.shape[0], x.shape[1] * 2, x.shape[2] // 2, x.shape[3], x.shape[4])
        x_4 = x.shape[3] // 4
        y_4 = x.shape[4] // 4
        return x[:, :, :, x_4 : 3 * x_4, y_4 : 3 * y_4]
