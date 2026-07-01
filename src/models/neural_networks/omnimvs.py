import torch
import torch.nn.functional as F

from src.models.basic import Conv2D, Conv3D, DeConv3D


class FeatureLayers(torch.nn.Module):
    def __init__(self, ch=32, use_rgb=False):
        super().__init__()
        in_ch  = 3 if use_rgb else 1
        layers = [Conv2D(in_ch, ch, 5, 2, 2)]
        layers += [Conv2D(ch, ch, 3, 1, 1) for _ in range(10)]
        for d in range(2, 5):
            layers += [Conv2D(ch, ch, 3, 1, d, dilation=d) for _ in range(2)]
        layers.append(Conv2D(ch, ch, 3, 1, 1, use_bn=False, relu=False))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1, 17, 2):
            x_ = self.layers[i](x)
            x  = self.layers[i + 1](x_, residual=x)
        return self.layers[17](x)


class SphericalSweep(torch.nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.transfer_conv = Conv2D(ch, ch, 3, 2, 1, use_bn=False, relu=False)

    def forward(self, feature, grid):
        num_invdepth = grid.shape[0]
        sweep = [
            F.grid_sample(feature, grid[d].unsqueeze(0), align_corners=True)
            for d in range(num_invdepth)
        ]
        return self.transfer_conv(torch.cat(sweep, 0))


class CostCompute(torch.nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        ch *= 2
        self.fusion = Conv3D(2 * ch, ch, 3, 1, 1)
        convs = [
            Conv3D(ch, ch, 3, 1, 1),
            Conv3D(ch, ch, 3, 1, 1),
            Conv3D(ch, ch, 3, 1, 1),
            Conv3D(ch, 2 * ch, 3, 2, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 2 * ch, 3, 2, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 2 * ch, 3, 2, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 2 * ch, 3, 1, 1),
            Conv3D(2 * ch, 4 * ch, 3, 2, 1),
            Conv3D(4 * ch, 4 * ch, 3, 1, 1),
            Conv3D(4 * ch, 4 * ch, 3, 1, 1),
        ]
        self.convs   = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(4 * ch, 2 * ch, 3, 2, 1, out_pad=1)
        self.deconv2 = DeConv3D(2 * ch, 2 * ch, 3, 2, 1, out_pad=1)
        self.deconv3 = DeConv3D(2 * ch, 2 * ch, 3, 2, 1, out_pad=1)
        self.deconv4 = DeConv3D(2 * ch,     ch, 3, 2, 1, out_pad=1)
        self.deconv5 = DeConv3D(    ch,      1, 3, 2, 1, out_pad=1, use_bn=False, relu=False)

    def forward(self, feats):
        c  = self.fusion(feats)
        c  = self.convs[0](c)
        c1 = self.convs[2](self.convs[1](c))
        c  = self.convs[3](c)
        c2 = self.convs[5](self.convs[4](c))
        c  = self.convs[6](c)
        c3 = self.convs[8](self.convs[7](c))
        c  = self.convs[9](c)
        c4 = self.convs[11](self.convs[10](c))
        c  = self.convs[12](c)
        c5 = self.convs[14](self.convs[13](c))
        c  = self.deconv1(c5, residual=c4)
        c  = self.deconv2(c,  residual=c3)
        c  = self.deconv3(c,  residual=c2)
        c  = self.deconv4(c,  residual=c1)
        return self.deconv5(c)


class OmniMVS(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.CH            = config.parameters.model.channel
        self.num_invdepth  = config.parameters.spherical_sweep.num_invdepth
        device_str         = str(config.parameters.general.device) if torch.cuda.is_available() else 'cpu'

        self.feature_layers = FeatureLayers(self.CH, use_rgb=False)
        self.spherical_sweep = SphericalSweep(self.CH)
        self.cost_computes   = CostCompute(self.CH)
        self.disparity = (
            torch.arange(0, self.num_invdepth, requires_grad=False)
            .view((1, -1, 1, 1)).float().to(device_str)
        )

    def forward(self, images, grids, upsample=False, out_cost=False):
        feats           = [self.feature_layers(x) for x in images]
        spherical_feats = torch.cat(
            [self.spherical_sweep(feats[i], grids[i]) for i in range(len(images))], 1
        ).permute([1, 0, 2, 3])
        spherical_feats = torch.unsqueeze(spherical_feats, 0)
        costs = self.cost_computes(spherical_feats)

        if upsample:
            costs = F.interpolate(costs.squeeze(1), scale_factor=2,
                                  mode='bilinear', align_corners=True)
        else:
            costs = torch.squeeze(costs, 1)

        prob = F.softmax(costs, dim=1)
        disp = torch.sum(torch.mul(prob, self.disparity), 1)

        if out_cost:
            return disp, prob.squeeze(), costs.squeeze()
        return disp
