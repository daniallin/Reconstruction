import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mtn.base import BatchNorm, initial_weight
from utils.params import set_params


def _conv_layer(in_chan, out_chan, k, s=1, p=1, sync_bn=False):
    conv_block = nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=k, stride=s, padding=p, bias=False),
        BatchNorm(out_chan, sync_bn),
        nn.ReLU(inplace=True), )
    return conv_block


class Attention(nn.Module):
    def __init__(self, in_chans, out_chans, method):
        super(Attention, self).__init__()
        self.method = method
        # Define layers
        if self.method == 'general':
            self.attention = nn.Conv2d(in_chans, out_chans, 3, padding=1)
        elif self.method == 'concat':
            self.attention = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_chans, out_chans, 3, padding=1))

    def forward(self, x1, x2):
        energy = self._score(x1, x2)
        return F.softmax(energy)

    def _score(self, x1, x2):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = torch.bmm(x1.unsqueeze(1), x2.unsqueeze(2))
        elif self.method == 'general':
            energy = self.attention(x2)
            energy = torch.bmm(x1.unsqueeze(1), energy.unsqueeze(2))
        elif self.method == 'concat':
            energy = self.attention(torch.cat((x1, x2), dim=1))
        return energy


class DepthUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, low_size):
        super(DepthUpSample, self).__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_size+256, low_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(low_size, low_size, 3, padding=1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(out_channel, out_channel, 3, padding=1),
                                  nn.LeakyReLU(0.2))

    def forward(self, x, low_feature, seg_low):
        up_x = F.interpolate(x, size=low_feature.size()[2:], mode='bilinear', align_corners=True)
        low_feature = self.low_conv(torch.cat((low_feature, seg_low), dim=1))
        output = self.conv(torch.cat((up_x, low_feature), dim=1))
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
        up_x = F.interpolate(input, size=seg_low_out.size()[2:], mode='bilinear', align_corners=True)
        output = torch.cat((up_x, seg_low_out), dim=1)
        output = self.seg_conv(output)
        return output


class SegModel(nn.Module):
    def __init__(self, low_feature_size):
        super(SegModel, self).__init__()
        # semantic segmentation
        self.seg_up1 = SegUpSample(low_feature_size[0], 64, args.sync_bn)
        self.seg_up2 = SegUpSample(low_feature_size[1], 64, args.sync_bn)
        self.seg_up3 = SegUpSample(low_feature_size[2], 64, args.sync_bn)
        self.seg_last = nn.Conv2d(256, args.class_num, kernel_size=1, stride=1)

    def forward(self, x, low_level_features):
        # semantic segmentation
        seg_1 = self.seg_up1(x, low_level_features[0])
        seg_2 = self.seg_up2(seg_1, low_level_features[1])
        seg_3 = self.seg_up3(seg_2, low_level_features[2])
        seg_out = self.seg_last(seg_3)
        return seg_out, (seg_1, seg_2, seg_3)


class DepthModel(nn.Module):
    def __init__(self, chan, low_feature_sizes):
        super(DepthModel, self).__init__()
        # depth estimation
        self.depth_conv1 = nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=1)
        self.depth_up1 = DepthUpSample(chan // 1 + low_feature_sizes[0], chan // 2, low_feature_sizes[0])
        self.depth_up2 = DepthUpSample(chan // 2 + low_feature_sizes[1], chan // 4, low_feature_sizes[1])
        self.depth_up3 = DepthUpSample(chan // 4 + low_feature_sizes[2], chan // 8, low_feature_sizes[2])
        self.depth_conv2 = nn.Conv2d(chan // 8, 1, kernel_size=3, stride=1, padding=1)
        self.depth_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, low_level_features, seg_low):
        # depth estimation
        depth_x = self.depth_conv1(x)
        depth_1 = self.depth_up1(depth_x, low_level_features[0], seg_low[0])
        depth_2 = self.depth_up2(depth_1, low_level_features[1], seg_low[1])
        depth_3 = self.depth_up3(depth_2, low_level_features[2], seg_low[2])
        depth_out = self.depth_relu2(self.depth_conv2(depth_3))
        return depth_out, (depth_1, depth_2, depth_3)


class OdometryModel(nn.Module):
    def __init__(self, chan, img_size):
        super(OdometryModel, self).__init__()
        self.low_pool = nn.MaxPool2d(2, 2)
        self.attention1 = Attention(512, 256, method='concat')
        self.attention2 = Attention(384, 128, method='concat')
        # vision odometry
        vo_chan = 2
        self.vo_conv = nn.Conv2d(chan, vo_chan, kernel_size=1, stride=1)
        in_size = vo_chan * img_size[0] * img_size[1] / args.output_scale / args.output_scale
        # print("in_size is {}".format(in_size))
        self.lstm = nn.LSTM(
                    input_size=int(in_size),
                    hidden_size=args.rnn_hidden_size,
                    num_layers=2,
                    dropout=args.rnn_dropout_between,
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(args.rnn_dropout_out)
        self.linear = nn.Linear(in_features=args.rnn_hidden_size, out_features=6)

    def forward(self, x, low_feature1, low_feature2, batch_size, seq_len):
        low_feature1 = self.low_pool(low_feature1)
        low_feature2 = self.low_pool(low_feature2)
        low_feature1 = torch.mul(low_feature1, self.attention1(x, low_feature1))
        low_feature2 = torch.mul(low_feature2, self.attention2(x, low_feature2))
        # vision odometry
        vo_x = self.vo_conv(torch.cat((low_feature1, low_feature2), dim=1))
        vo_x = vo_x.view(batch_size, seq_len, -1)
        vo_out, hc = self.lstm(vo_x)
        vo_out = self.rnn_drop_out(vo_out)
        vo_out = self.linear(vo_out)
        return vo_out


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        img_size = args.crop_size
        if args.backbone == 'resnext':
            self.low_feature_sizes = [512, 256, 64]
            self.num_channels = 256
        else:
            raise NotImplementedError
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        self.seg_decoder = SegModel(self.low_feature_sizes)

        chan = int(self.num_channels)
        self.depth_decoder = DepthModel(chan, self.low_feature_sizes)

        self.vo_decoder = OdometryModel(int(chan*1.5), img_size)

        initial_weight(self.modules())

    def forward(self, x, low_level_features, seq_len):
        # semantic segmentation
        seg_out, seg_low = self.seg_decoder(x, low_level_features)

        # depth estimation
        depth_out, depth_low = self.depth_decoder(x, low_level_features, seg_low)

        # vision odometry
        vo_out = self.vo_decoder(x, seg_low[0], depth_low[0], self.args.batch_size, seq_len)

        return [depth_out, seg_out, vo_out], self.logsigma


if __name__ == '__main__':
    args = set_params()
    model = Decoder(args)
    model.eval()
    x = torch.randn(1, 256, 20, 30)
    low0 = torch.randn(1, 64, 160, 240)
    low1 = torch.randn(1, 256, 80, 120)
    low2 = torch.randn(1, 512, 40, 60)
    y = model(x, [low2, low1, low0], 1)
    print(y[0][0].size())
    print(y[0][1].size())
    print(y[0][2].size())
