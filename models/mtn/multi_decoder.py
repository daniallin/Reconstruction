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
    def __init__(self, pedestrian_num, hidden_size, method, use_cuda, bilstm):
        super(Attention, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.hidden_size = hidden_size
        self.method = method
        self.use_cuda = use_cuda
        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size * bilstm, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.ModuleList([
                nn.Linear(self.hidden_size * 2 * bilstm, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            ])

    def forward(self, decoder_hiddens, encoder_outputs):
        batch_size = encoder_outputs.size()[0]
        energies = torch.zeros(batch_size, self.pedestrian_num)
        energies = energies.cuda() if self.use_cuda else energies
        for i in range(self.pedestrian_num):
            energies[:, i] = self._score(decoder_hiddens[:, i, :], encoder_outputs[:, i, :])
        return F.softmax(energies, dim=-1)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = torch.bmm(hidden.unsqueeze(1), encoder_output.unsqueeze(2)).squeeze(-1)
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2)).squeeze(-1)
        elif self.method == 'concat':
            energy = self.attention[0](torch.cat((hidden, encoder_output), -1))
            energy = self.attention[1](energy)
            energy = self.attention[2](energy)
        return energy.squeeze(-1)


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
        seg_x = self.seg_up1(x, low_level_features[0])
        seg_x = self.seg_up2(seg_x, low_level_features[1])
        seg_x = self.seg_up3(seg_x, low_level_features[2])
        seg_out = self.seg_last(seg_x)
        return seg_out


class DepthModel(nn.Module):
    def __init__(self, chan):
        super(DepthModel, self).__init__()
        # depth estimation
        self.depth_conv1 = nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=1)
        self.depth_up1 = DepthUpSample(chan // 1 + 512, chan // 2)
        self.depth_up2 = DepthUpSample(chan // 2 + 256, chan // 4)
        self.depth_up3 = DepthUpSample(chan // 4 + 64, chan // 8)
        self.depth_conv2 = nn.Conv2d(chan // 8, 1, kernel_size=3, stride=1, padding=1)
        self.depth_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, low_level_features):
        # depth estimation
        depth_x = self.depth_conv1(x)
        depth_x = self.depth_up1([depth_x, low_level_features[0]])
        depth_x = self.depth_up2([depth_x, low_level_features[1]])
        depth_x = self.depth_up3([depth_x, low_level_features[2]])
        depth_out = self.depth_relu2(self.depth_conv2(depth_x))
        return depth_out


class OdometryModel(nn.Module):
    def __init__(self, chan, img_size):
        super(OdometryModel, self).__init__()

        # vision odometry
        vo_chan = 2
        self.vo_conv = nn.Conv2d(chan, vo_chan, kernel_size=1, stride=1)
        in_size = vo_chan * img_size[0] * img_size[1] / args.output_scale / args.output_scale
        # print("in_size is {}".format(in_size))
        self.rnn = nn.LSTM(
                    input_size=int(in_size),
                    hidden_size=args.rnn_hidden_size,
                    num_layers=2,
                    dropout=args.rnn_dropout_between,
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(args.rnn_dropout_out)
        self.linear = nn.Linear(in_features=args.rnn_hidden_size, out_features=6)

    def forward(self, x, low_level_features, batch_size, seq_len):
        # vision odometry
        vo_x = self.vo_conv(x)
        vo_x = vo_x.view(batch_size, seq_len, -1)
        vo_out, hc = self.rnn(vo_x)
        vo_out = self.rnn_drop_out(vo_out)
        vo_out = self.linear(vo_out)
        return vo_out


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        img_size = args.crop_size
        if args.backbone == 'resnext':
            self.low_feature_size = [512, 256, 64]
            self.num_channels = 256
        else:
            raise NotImplementedError
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        self.seg_decoder = SegModel(self.low_feature_size)

        chan = int(self.num_channels)
        self.depth_decoder = DepthModel(chan)

        self.vo_decoder = OdometryModel(chan, img_size)

        initial_weight(self.modules())

    def forward(self, x, low_level_features, seq_len):
        # semantic segmentation
        seg_out = self.seg_decoder(x, low_level_features)

        # depth estimation
        depth_out = self.depth_decoder(x, low_level_features)

        # vision odometry
        vo_out = self.vo_decoder(x, low_level_features, self.args.batch_size, seq_len)

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
