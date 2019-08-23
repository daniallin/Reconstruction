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
        self.decoder = Decoder(args)

        if args.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        # input: batch_size, seq_;en, channel, height, width
        # input = torch.cat((input[:, :-1], input[:, 1:]), dim=2)
        # print(input.size())
        batch_size = input.size(0)
        seq_len = input.size(1)
        input = input.view(batch_size*seq_len, input.size(2), input.size(3), input.size(4))

        output, low_level_feature = self.encoder(input)
        output = self.aspp(output)
        output, logsigma = self.decoder(output, low_level_feature, seq_len)
        # print(output[0].size())

        for i in range(len(output)-1):
            output[i] = F.interpolate(output[i], size=input.size()[2:], mode='bilinear', align_corners=True)
            out_size = output[i].size()
            print(out_size)
            output[i] = output[i].view(batch_size, seq_len, out_size[1], out_size[2], out_size[3])

        return output, logsigma


if __name__ == '__main__':
    model = ReconstructMTN()
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output[0].size())




