from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch

# GBN block
class W_bn(nn.Module):
    def __init__(self, channel, feature_shape, is_conv):  # feature_shape: [C, H, W]
        super(W_bn, self).__init__()
        # four bn branches
        self.bn = nn.BatchNorm2d(channel)
        self.bn_l1 = nn.BatchNorm2d(channel)
        self.bn_l2 = nn.BatchNorm2d(channel)
        self.bn_linf = nn.BatchNorm2d(channel)
        self.four_bns = [self.bn, self.bn_l1, self.bn_l2, self.bn_linf]
        self.is_conv = is_conv
        if is_conv:
            # conv gate
            self.fc_conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc_conv2 = nn.Conv2d(channel, channel*2, kernel_size=3, stride=2, padding=1, bias=False)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(channel*2 * int(feature_shape[1]/2) * int(feature_shape[2]/2), 4)
        else:
            # gate
            self.fc1 = nn.Linear(feature_shape[0] * feature_shape[1] * feature_shape[2], 512)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x, is_training, norm_type=None):
        # x: (N, C, H, W)
        # norm_type: 0-clean, 1-L1, 2-L2, 3-Linf
        if is_training:  # training phase
            if self.is_conv:
                gate_out = self.fc_conv1(x)
                gate_out = self.relu1(gate_out)
                gate_out = self.fc_conv2(gate_out)
                gate_out = self.relu2(gate_out)
                gate_out = self.fc(gate_out.view(gate_out.size(0), -1))
            else:
                gate_out = self.fc1(x.view(x.size(0), -1))
                gate_out = self.relu1(gate_out)
                gate_out = self.fc2(gate_out)
            gate_loss = F.cross_entropy(gate_out, torch.full([x.size(0)], norm_type).long().cuda())
            bn_out = self.four_bns[norm_type](x)
            return bn_out, gate_loss
        else:  # testing phase
            out_clean = self.four_bns[0](x)
            out_l1 = self.four_bns[1](x)
            out_l2 = self.four_bns[2](x)
            out_linf = self.four_bns[3](x)
            # gate forward
            if self.is_conv:
                gate_out = self.fc_conv1(x)
                gate_out = self.relu1(gate_out)
                gate_out = self.fc_conv2(gate_out)
                gate_out = self.relu2(gate_out)
                gate_out = self.fc(gate_out.view(gate_out.size(0), -1))
            else:
                gate_out = self.fc1(x.view(x.size(0), -1))
                gate_out = self.relu1(gate_out)
                gate_out = self.fc2(gate_out)
            weight = F.softmax(gate_out)  # None, 4
            # weighted bn
            w0 = weight[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.shape[1], x.shape[2], x.shape[3])
            w1 = weight[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.shape[1], x.shape[2], x.shape[3])
            w2 = weight[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.shape[1], x.shape[2], x.shape[3])
            w3 = weight[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, x.shape[1], x.shape[2], x.shape[3])
            out_w_bn = w0 * out_clean + w1 * out_l1 + w2 * out_l2 + w3 * out_linf
            return out_w_bn
        

# lenet5 with GBN module
class lenet5(Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.linear3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x, is_training=False, norm_type=0):
        ret_gate_loss = 0
        
        y = self.conv1(x)
        if is_training:
            if isinstance(self.bn1, nn.BatchNorm2d):
                y = y
            else:
                y, gate_loss = self.bn1(y, is_training, norm_type)
                ret_gate_loss = ret_gate_loss + gate_loss
        else:
            if isinstance(self.bn1, nn.BatchNorm2d):
                y = y
            else:
                y = self.bn1(y, is_training, norm_type)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        if is_training:
            if isinstance(self.bn2, nn.BatchNorm2d):
                y = y
            else:
                y, gate_loss = self.bn2(y, is_training, norm_type)
                ret_gate_loss = ret_gate_loss + gate_loss
        else:
            if isinstance(self.bn2, nn.BatchNorm2d):
                y = y
            else:
                y = self.bn2(y, is_training, norm_type)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.linear1(y)
        y = self.relu3(y)
        y = self.linear2(y)
        y = self.relu4(y)
        y = self.linear3(y)
        y = self.relu5(y)

        if is_training is True:
            return y, ret_gate_loss
        else:
            return y


def lenet5_gate1_conv():
    model = lenet5()
    # add gate bn
    model.bn1 = W_bn(6, [6, 24, 24], True)
    return model

def lenet5_gate2_conv():
    model = lenet5()
    # add gate bn
    model.bn2 = W_bn(16, [16, 8, 8], True)
    return model

def lenet5_conv1_fc():
    model = lenet5()
    # add gate bn
    model.bn1 = W_bn(6, [6, 24, 24], True)
    model.bn2 = W_bn(16, [16, 8, 8], False)
    return model
