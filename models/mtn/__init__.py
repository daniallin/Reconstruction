import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mtn.resnext import resnext101_32x8d
from models.mtn.aspp import ASPP
from models.mtn.multi_decoder import Decoder


class ReconstructMTN(nn.Module):
    def __init__(self, args):
        super(ReconstructMTN, self).__init__()
        self.encoder = resnext101_32x8d(args.use_pretrain, replace_stride_with_dilation=[False, False, True])
        self.aspp = ASPP(args)
        self.decoder = Decoder(args, args.crop_size)

        if args.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        output, low_level_feature = self.encoder(input)
        output = self.aspp(output)
        output = self.decoder(output, low_level_feature)
        # print(output[0].size())
        for i in range(len(output)-1):
            output[i] = F.interpolate(output[i], size=input.size()[2:], mode='bilinear', align_corners=True)

        return output


if __name__ == '__main__':
    model = ReconstructMTN()
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output[0].size())




