import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mtn.resnext import resnext101_32x8d
from models.mtn.aspp import ASPP
from models.mtn.multi_decoder import Decoder


class ReconstructMTN(nn.Module):
    def __init__(self, backbone='resnext', sync_bn=False, num_classes=10,
                 freeze_bn=False, output_scale=16, pretrained=False):
        super(ReconstructMTN, self).__init__()
        if backbone == 'drn':
            output_scale = 8

        self.encoder = resnext101_32x8d(pretrained, replace_stride_with_dilation=[False, False, True])
        self.aspp = ASPP(backbone, output_scale, sync_bn)
        self.decoder = Decoder(backbone, num_classes, sync_bn)

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        output, low_level_feature = self.encoder(input)
        output = self.aspp(output)
        output = self.decoder(output, low_level_feature)
        print(output[0].size())
        for i in range(len(output)):
            output[i] = F.interpolate(output[i], size=input.size()[2:], mode='bilinear', align_corners=True)

        return F.interpolate(output[i], size=input.size()[2:], mode='bilinear', align_corners=True)


if __name__ == '__main__':
    model = ReconstructMTN()
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())




