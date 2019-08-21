import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mtn.base import BatchNorm, initial_weight


class ASPPBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, padding, dilation, sync_bn=False):
        super(ASPPBlock, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=1,
                                     padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes, sync_bn)
        # what will happen if use LeakyReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, args):
        super(ASPP, self).__init__()
        if args.backbone == 'resnext':
            inplanes = 2048
        else:
            inplanes = 2048
        if args.output_scale == 16:
            dilations = [1, 6, 12, 18]
        elif args.output_scale == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.layer1 = ASPPBlock(inplanes, 256, 1, padding=0, dilation=dilations[0], sync_bn=sync_bn)
        self.layer2 = ASPPBlock(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], sync_bn=sync_bn)
        self.layer3 = ASPPBlock(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], sync_bn=sync_bn)
        self.layer4 = ASPPBlock(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], sync_bn=sync_bn)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, 1, bias=False),
                                             BatchNorm(256, args.sync_bn),
                                             nn.ReLU())
        self.last_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                       BatchNorm(256, args.sync_bn),
                                       nn.ReLU())
        self.dropout = nn.Dropout(0.5)

        initial_weight(self.modules())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.last_conv(x)

        return self.dropout(x)


if __name__ == '__main__':
    model = ASPP(backbone='resnext', sync_bn=False, output_scale=16)
    model.eval()
    # when model.train(), the batch size should be > 1,
    # https://stackoverflow.com/questions/48343857/whats-the-reason-of-the-error-valueerror-expected-more-than-1-value-per-channe
    x = torch.rand(1, 2048, 5, 5)
    output = model(x)
    print(output.size())
