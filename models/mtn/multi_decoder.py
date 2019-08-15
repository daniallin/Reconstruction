import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import BatchNorm, initial_weight
from utils.params import set_params


def _conv_layer(in_chan, out_chan, k, s=1, p=1, sync_bn=False):
    conv_block = nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=k, stride=s, padding=p, bias=False),
        BatchNorm(out_chan, sync_bn),
        nn.ReLU(inplace=True), )
    return conv_block


class DepthUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthUpSample, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(out_channel, out_channel, 3, padding=1),
                                  nn.LeakyReLU(0.2))

    def forward(self, input):
        up_x = F.interpolate(input[0], size=input[1].size()[2:], mode='bilinear', align_corners=True)
        output = self.conv(torch.cat((up_x, input[1]), dim=1))
        return output


class SegUpSample(nn.Module):
    def __init__(self, in_channel, out_channel=64, sync_bn=False):
        super(SegUpSample, self).__init__()
        self.seg_low_conv = _conv_layer(in_channel, out_channel, k=1, p=0, sync_bn=sync_bn)
        self.seg_conv = nn.Sequential(_conv_layer(out_channel+256, 256, k=3, s=1, p=1, sync_bn=sync_bn),
                                      nn.Dropout(0.5),
                                      _conv_layer(256, 256, k=3, s=1, p=1, sync_bn=sync_bn),
                                      nn.Dropout(0.1))

    def forward(self, input, low_feature):
        seg_low_out = self.seg_low_conv(low_feature)
        up_x = F.interpolate(x, size=seg_low_out.size()[2:], mode='bilinear', align_corners=True)
        output = torch.cat((up_x, seg_low_out), dim=1)
        output = self.seg_conv(output)
        return output


class Decoder(nn.Module):
    def __init__(self, args, img_size):
        super(Decoder, self).__init__()
        self.args = args
        if args.backbone == 'resnext':
            self.low_feature_size = [512, 256, 64]
            self.num_channels = 256
        else:
            raise NotImplementedError

        # semantic segmentation
        self.seg_up1 = SegUpSample(self.low_feature_size[0], 64, args.sync_bn)
        self.seg_up2 = SegUpSample(self.low_feature_size[1], 64, args.sync_bn)
        self.seg_up3 = SegUpSample(self.low_feature_size[2], 64, args.sync_bn)
        self.seg_last = nn.Conv2d(256, args.class_num, kernel_size=1, stride=1)

        # depth estimation
        chan = int(self.num_channels)
        self.depth_conv1 = nn.Conv2d(self.num_channels, chan, kernel_size=1, stride=1, padding=1)
        self.depth_up1 = DepthUpSample(chan // 1 + 512, chan // 2)
        self.depth_up2 = DepthUpSample(chan // 2 + 256, chan // 4)
        self.depth_up3 = DepthUpSample(chan // 4 + 64, chan // 8)
        self.depth_conv2 = nn.Conv2d(chan // 8, 1, kernel_size=3, stride=1, padding=1)
        self.depth_relu2 = nn.LeakyReLU(0.2)

        # vision odometry
        vo_chan = 2
        self.vo_conv = nn.Conv2d(256, vo_chan, kernel_size=1, stride=1)
        in_size = vo_chan * img_size[0] * img_size[1] / args.output_scale / args.output_scale
        self.rnn = nn.LSTM(
                    input_size=int(in_size),
                    hidden_size=args.rnn_hidden_size,
                    num_layers=2,
                    dropout=args.rnn_dropout_between,
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(args.rnn_dropout_out)
        self.linear = nn.Linear(in_features=args.rnn_hidden_size, out_features=6)

        initial_weight(self.modules())

    def forward(self, x, low_level_features):
        # semantic segmentation
        seg_x = self.seg_up1(x, low_level_features[0])
        seg_x = self.seg_up2(seg_x, low_level_features[1])
        seg_x = self.seg_up3(seg_x, low_level_features[2])
        seg_out = self.seg_last(seg_x)

        # depth estimation
        depth_x = self.depth_conv1(x)
        depth_x = self.depth_up1([depth_x, low_level_features[0]])
        depth_x = self.depth_up2([depth_x, low_level_features[1]])
        depth_x = self.depth_up3([depth_x, low_level_features[2]])
        depth_out = self.depth_relu2(self.depth_conv2(depth_x))

        # vision odometry
        vo_x = self.vo_conv(x)
        vo_x = vo_x.view(self.args.batch_size, self.args.seq_len, -1)
        vo_out, hc = self.rnn(vo_x)
        vo_out = self.rnn_drop_out(vo_out)
        vo_out = self.linear(vo_out)

        return [seg_out, depth_out, vo_out]


if __name__ == '__main__':
    args = set_params()
    model = Decoder(args, np.array((512, 512)))
    model.eval()
    x = torch.randn(2, 256, 32, 32)
    low0 = torch.randn(2, 64, 256, 256)
    low1 = torch.randn(2, 256, 128, 128)
    low2 = torch.randn(2, 512, 64, 64)
    y = model(x, [low2, low1, low0])
    print(y[0].size())
    print(y[1].size())
    print(y[2].size())
