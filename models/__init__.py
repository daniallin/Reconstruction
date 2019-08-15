import torch.nn as nn
from models.sync_batchnorm import SynchronizedBatchNorm2d


def BatchNorm(planes, sync_bn=False):
    if not sync_bn:
        return SynchronizedBatchNorm2d(planes)
    return nn.BatchNorm2d(planes)


def initial_weight(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # layer 1
            nn.init.kaiming_normal_(m.weight_ih_l0)  # orthogonal_(m.weight_ih_l0)
            nn.init.kaiming_normal_(m.weight_hh_l0)
            m.bias_ih_l0.data.zero_()
            m.bias_hh_l0.data.zero_()
            # Set forget gate bias to 1 (remember)
            n = m.bias_hh_l0.size(0)
            start, end = n // 4, n // 2
            m.bias_hh_l0.data[start:end].fill_(1.)

            # layer 2
            nn.init.kaiming_normal_(m.weight_ih_l1)  # orthogonal_(m.weight_ih_l1)
            nn.init.kaiming_normal_(m.weight_hh_l1)
            m.bias_ih_l1.data.zero_()
            m.bias_hh_l1.data.zero_()
            n = m.bias_hh_l1.size(0)
            start, end = n // 4, n // 2
            m.bias_hh_l1.data[start:end].fill_(1.)
