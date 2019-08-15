import torch
import torch.nn as nn
import torch.nn.functional as F

from models import BatchNorm, initial_weight


class Decoder(nn.Module):
    def __init__(self, backbone, num_classes, sync_bn=False):
        super(Decoder, self).__init__()
        if backbone == 'resnext':
            low_feature_size = [64, 256, 512]
        else:
            raise NotImplementedError
        filter = [64, 256, 512]

        conv_low_features = nn.ModuleList()
        for i in range(len(low_feature_size)):
            conv_low_features.append(self._conv_layer(low_feature_size[i], filter[0], k=1, p=0, sync_bn=sync_bn))

        self.seg_conv1 = self._conv_layer(low_feature_size[1], filter[0], k=1, p=0, sync_bn=sync_bn)
        # here 320 = 256 + 64, is the sum size of low level feature and output feature
        self.seg_conv2 = nn.Sequential(self._conv_layer(low_feature_size[1]+filter[0], filter[1], k=3, s=1, p=1, sync_bn=sync_bn),
                                       nn.Dropout(0.5),
                                       self._conv_layer(filter[1], filter[1], k=3, s=1, p=1, sync_bn=sync_bn),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(filter[1], num_classes, kernel_size=1, stride=1))

        self.depth_conv = conv_low_features
        self.vo_conv = conv_low_features
        initial_weight(self.modules())

    def _conv_layer(self, in_chan, out_chan, k, s=1, p=1, sync_bn=False):
        conv_block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=k, stride=s, padding=p, bias=False),
            BatchNorm(out_chan, sync_bn),
            nn.ReLU(inplace=True),)
        return conv_block

    def forward(self, x, low_level_feature):
        # semantic segmentation
        seg_low_out = self.seg_conv1(low_level_feature[1])
        seg_x = F.interpolate(x, size=seg_low_out.size()[2:], mode='bilinear', align_corners=True)
        seg_x = torch.cat((seg_x, seg_low_out), dim=1)
        seg_out = self.seg_conv2(seg_x)

        return [seg_out, seg_out]


if __name__ == '__main__':
    model = Decoder(backbone='resnext', num_classes=10)
    model.eval()
    x = torch.randn(1, 256, 32, 32)
    low = torch.randn(1, 256, 64, 64)
    y = model(x, [x, low])
    print(y.size())
