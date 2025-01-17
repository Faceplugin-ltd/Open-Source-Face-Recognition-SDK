import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class irn50_pytorch(nn.Module):
    def __init__(self, weight_file):
        super(irn50_pytorch, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.Convolution1 = self.__conv(2, name='Convolution1', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.BatchNorm1 = self.__batch_normalization(2, 'BatchNorm1', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution2 = self.__conv(2, name='Convolution2', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm2 = self.__batch_normalization(2, 'BatchNorm2', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution3 = self.__conv(2, name='Convolution3', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm3 = self.__batch_normalization(2, 'BatchNorm3', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution4 = self.__conv(2, name='Convolution4', in_channels=64, out_channels=80, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm4 = self.__batch_normalization(2, 'BatchNorm4', num_features=80, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution5 = self.__conv(2, name='Convolution5', in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm5 = self.__batch_normalization(2, 'BatchNorm5', num_features=192, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution6 = self.__conv(2, name='Convolution6', in_channels=192, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.BatchNorm6 = self.__batch_normalization(2, 'BatchNorm6', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res1_proj = self.__conv(2, name='conv2_res1_proj', in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res1_conv1 = self.__conv(2, name='conv2_res1_conv1', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res1_conv1_bn = self.__batch_normalization(2, 'conv2_res1_conv1_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res1_conv2 = self.__conv(2, name='conv2_res1_conv2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2_res1_conv2_bn = self.__batch_normalization(2, 'conv2_res1_conv2_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res1_conv3 = self.__conv(2, name='conv2_res1_conv3', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res2_pre_bn = self.__batch_normalization(2, 'conv2_res2_pre_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res2_conv1 = self.__conv(2, name='conv2_res2_conv1', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res2_conv1_bn = self.__batch_normalization(2, 'conv2_res2_conv1_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res2_conv2 = self.__conv(2, name='conv2_res2_conv2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2_res2_conv2_bn = self.__batch_normalization(2, 'conv2_res2_conv2_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res2_conv3 = self.__conv(2, name='conv2_res2_conv3', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res3_pre_bn = self.__batch_normalization(2, 'conv2_res3_pre_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res3_conv1 = self.__conv(2, name='conv2_res3_conv1', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_res3_conv1_bn = self.__batch_normalization(2, 'conv2_res3_conv1_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res3_conv2 = self.__conv(2, name='conv2_res3_conv2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv2_res3_conv2_bn = self.__batch_normalization(2, 'conv2_res3_conv2_bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_res3_conv3 = self.__conv(2, name='conv2_res3_conv3', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res1_pre_bn = self.__batch_normalization(2, 'conv3_res1_pre_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res1_proj = self.__conv(2, name='conv3_res1_proj', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.conv3_res1_conv1 = self.__conv(2, name='conv3_res1_conv1', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.conv3_res1_conv1_bn = self.__batch_normalization(2, 'conv3_res1_conv1_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res1_conv2 = self.__conv(2, name='conv3_res1_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv3_res1_conv2_bn = self.__batch_normalization(2, 'conv3_res1_conv2_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res1_conv3 = self.__conv(2, name='conv3_res1_conv3', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res2_pre_bn = self.__batch_normalization(2, 'conv3_res2_pre_bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res2_conv1 = self.__conv(2, name='conv3_res2_conv1', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res2_conv1_bn = self.__batch_normalization(2, 'conv3_res2_conv1_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res2_conv2 = self.__conv(2, name='conv3_res2_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv3_res2_conv2_bn = self.__batch_normalization(2, 'conv3_res2_conv2_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res2_conv3 = self.__conv(2, name='conv3_res2_conv3', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res3_pre_bn = self.__batch_normalization(2, 'conv3_res3_pre_bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res3_conv1 = self.__conv(2, name='conv3_res3_conv1', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res3_conv1_bn = self.__batch_normalization(2, 'conv3_res3_conv1_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res3_conv2 = self.__conv(2, name='conv3_res3_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv3_res3_conv2_bn = self.__batch_normalization(2, 'conv3_res3_conv2_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res3_conv3 = self.__conv(2, name='conv3_res3_conv3', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res4_pre_bn = self.__batch_normalization(2, 'conv3_res4_pre_bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res4_conv1 = self.__conv(2, name='conv3_res4_conv1', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_res4_conv1_bn = self.__batch_normalization(2, 'conv3_res4_conv1_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res4_conv2 = self.__conv(2, name='conv3_res4_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv3_res4_conv2_bn = self.__batch_normalization(2, 'conv3_res4_conv2_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_res4_conv3 = self.__conv(2, name='conv3_res4_conv3', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res1_pre_bn = self.__batch_normalization(2, 'conv4_res1_pre_bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res1_proj = self.__conv(2, name='conv4_res1_proj', in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res1_conv1 = self.__conv(2, name='conv4_res1_conv1', in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv4_res1_conv1_bn = self.__batch_normalization(2, 'conv4_res1_conv1_bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res1_conv2 = self.__conv(2, name='conv4_res1_conv2', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res2_pre_bn = self.__batch_normalization(2, 'conv4_res2_pre_bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res2_conv1_proj = self.__conv(2, name='conv4_res2_conv1_proj', in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res2_conv1 = self.__conv(2, name='conv4_res2_conv1', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res2_conv1_bn = self.__batch_normalization(2, 'conv4_res2_conv1_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res2_conv2 = self.__conv(2, name='conv4_res2_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv4_res2_conv2_bn = self.__batch_normalization(2, 'conv4_res2_conv2_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res2_conv3 = self.__conv(2, name='conv4_res2_conv3', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res3_pre_bn = self.__batch_normalization(2, 'conv4_res3_pre_bn', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res3_conv1 = self.__conv(2, name='conv4_res3_conv1', in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_res3_conv1_bn = self.__batch_normalization(2, 'conv4_res3_conv1_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res3_conv2 = self.__conv(2, name='conv4_res3_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv4_res3_conv2_bn = self.__batch_normalization(2, 'conv4_res3_conv2_bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_res3_conv3 = self.__conv(2, name='conv4_res3_conv3', in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv5_bn = self.__batch_normalization(2, 'conv5_bn', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.fc1_1 = self.__dense(name = 'fc1_1', in_features = 16384, out_features = 512, bias = False)
        self.bn_fc1 = self.__batch_normalization(1, 'bn_fc1', num_features=512, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        Convolution1    = self.Convolution1(x)
        BatchNorm1      = self.BatchNorm1(Convolution1)
        ReLU1           = F.relu(BatchNorm1)
        Convolution2    = self.Convolution2(ReLU1)
        BatchNorm2      = self.BatchNorm2(Convolution2)
        ReLU2           = F.relu(BatchNorm2)
        Convolution3_pad = F.pad(ReLU2, (1, 1, 1, 1))
        Convolution3    = self.Convolution3(Convolution3_pad)
        BatchNorm3      = self.BatchNorm3(Convolution3)
        ReLU3           = F.relu(BatchNorm3)
        Pooling1_pad    = F.pad(ReLU3, (0, 1, 0, 1), value=float('-inf'))
        Pooling1, Pooling1_idx = F.max_pool2d(Pooling1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        Convolution4    = self.Convolution4(Pooling1)
        BatchNorm4      = self.BatchNorm4(Convolution4)
        ReLU4           = F.relu(BatchNorm4)
        Convolution5    = self.Convolution5(ReLU4)
        BatchNorm5      = self.BatchNorm5(Convolution5)
        ReLU5           = F.relu(BatchNorm5)
        Convolution6_pad = F.pad(ReLU5, (1, 1, 1, 1))
        Convolution6    = self.Convolution6(Convolution6_pad)
        BatchNorm6      = self.BatchNorm6(Convolution6)
        ReLU6           = F.relu(BatchNorm6)
        conv2_res1_proj = self.conv2_res1_proj(ReLU6)
        conv2_res1_conv1 = self.conv2_res1_conv1(ReLU6)
        conv2_res1_conv1_bn = self.conv2_res1_conv1_bn(conv2_res1_conv1)
        conv2_res1_conv1_relu = F.relu(conv2_res1_conv1_bn)
        conv2_res1_conv2_pad = F.pad(conv2_res1_conv1_relu, (1, 1, 1, 1))
        conv2_res1_conv2 = self.conv2_res1_conv2(conv2_res1_conv2_pad)
        conv2_res1_conv2_bn = self.conv2_res1_conv2_bn(conv2_res1_conv2)
        conv2_res1_conv2_relu = F.relu(conv2_res1_conv2_bn)
        conv2_res1_conv3 = self.conv2_res1_conv3(conv2_res1_conv2_relu)
        conv2_res1      = conv2_res1_proj + conv2_res1_conv3
        conv2_res2_pre_bn = self.conv2_res2_pre_bn(conv2_res1)
        conv2_res2_pre_relu = F.relu(conv2_res2_pre_bn)
        conv2_res2_conv1 = self.conv2_res2_conv1(conv2_res2_pre_relu)
        conv2_res2_conv1_bn = self.conv2_res2_conv1_bn(conv2_res2_conv1)
        conv2_res2_conv1_relu = F.relu(conv2_res2_conv1_bn)
        conv2_res2_conv2_pad = F.pad(conv2_res2_conv1_relu, (1, 1, 1, 1))
        conv2_res2_conv2 = self.conv2_res2_conv2(conv2_res2_conv2_pad)
        conv2_res2_conv2_bn = self.conv2_res2_conv2_bn(conv2_res2_conv2)
        conv2_res2_conv2_relu = F.relu(conv2_res2_conv2_bn)
        conv2_res2_conv3 = self.conv2_res2_conv3(conv2_res2_conv2_relu)
        conv2_res2      = conv2_res1 + conv2_res2_conv3
        conv2_res3_pre_bn = self.conv2_res3_pre_bn(conv2_res2)
        conv2_res3_pre_relu = F.relu(conv2_res3_pre_bn)
        conv2_res3_conv1 = self.conv2_res3_conv1(conv2_res3_pre_relu)
        conv2_res3_conv1_bn = self.conv2_res3_conv1_bn(conv2_res3_conv1)
        conv2_res3_conv1_relu = F.relu(conv2_res3_conv1_bn)
        conv2_res3_conv2_pad = F.pad(conv2_res3_conv1_relu, (1, 1, 1, 1))
        conv2_res3_conv2 = self.conv2_res3_conv2(conv2_res3_conv2_pad)
        conv2_res3_conv2_bn = self.conv2_res3_conv2_bn(conv2_res3_conv2)
        conv2_res3_conv2_relu = F.relu(conv2_res3_conv2_bn)
        conv2_res3_conv3 = self.conv2_res3_conv3(conv2_res3_conv2_relu)
        conv2_res3      = conv2_res2 + conv2_res3_conv3
        conv3_res1_pre_bn = self.conv3_res1_pre_bn(conv2_res3)
        conv3_res1_pre_relu = F.relu(conv3_res1_pre_bn)
        conv3_res1_proj = self.conv3_res1_proj(conv3_res1_pre_relu)
        conv3_res1_conv1 = self.conv3_res1_conv1(conv3_res1_pre_relu)
        conv3_res1_conv1_bn = self.conv3_res1_conv1_bn(conv3_res1_conv1)
        conv3_res1_conv1_relu = F.relu(conv3_res1_conv1_bn)
        conv3_res1_conv2_pad = F.pad(conv3_res1_conv1_relu, (1, 1, 1, 1))
        conv3_res1_conv2 = self.conv3_res1_conv2(conv3_res1_conv2_pad)
        conv3_res1_conv2_bn = self.conv3_res1_conv2_bn(conv3_res1_conv2)
        conv3_res1_conv2_relu = F.relu(conv3_res1_conv2_bn)
        conv3_res1_conv3 = self.conv3_res1_conv3(conv3_res1_conv2_relu)
        conv3_res1      = conv3_res1_proj + conv3_res1_conv3
        conv3_res2_pre_bn = self.conv3_res2_pre_bn(conv3_res1)
        conv3_res2_pre_relu = F.relu(conv3_res2_pre_bn)
        conv3_res2_conv1 = self.conv3_res2_conv1(conv3_res2_pre_relu)
        conv3_res2_conv1_bn = self.conv3_res2_conv1_bn(conv3_res2_conv1)
        conv3_res2_conv1_relu = F.relu(conv3_res2_conv1_bn)
        conv3_res2_conv2_pad = F.pad(conv3_res2_conv1_relu, (1, 1, 1, 1))
        conv3_res2_conv2 = self.conv3_res2_conv2(conv3_res2_conv2_pad)
        conv3_res2_conv2_bn = self.conv3_res2_conv2_bn(conv3_res2_conv2)
        conv3_res2_conv2_relu = F.relu(conv3_res2_conv2_bn)
        conv3_res2_conv3 = self.conv3_res2_conv3(conv3_res2_conv2_relu)
        conv3_res2      = conv3_res1 + conv3_res2_conv3
        conv3_res3_pre_bn = self.conv3_res3_pre_bn(conv3_res2)
        conv3_res3_pre_relu = F.relu(conv3_res3_pre_bn)
        conv3_res3_conv1 = self.conv3_res3_conv1(conv3_res3_pre_relu)
        conv3_res3_conv1_bn = self.conv3_res3_conv1_bn(conv3_res3_conv1)
        conv3_res3_conv1_relu = F.relu(conv3_res3_conv1_bn)
        conv3_res3_conv2_pad = F.pad(conv3_res3_conv1_relu, (1, 1, 1, 1))
        conv3_res3_conv2 = self.conv3_res3_conv2(conv3_res3_conv2_pad)
        conv3_res3_conv2_bn = self.conv3_res3_conv2_bn(conv3_res3_conv2)
        conv3_res3_conv2_relu = F.relu(conv3_res3_conv2_bn)
        conv3_res3_conv3 = self.conv3_res3_conv3(conv3_res3_conv2_relu)
        conv3_res3      = conv3_res2 + conv3_res3_conv3
        conv3_res4_pre_bn = self.conv3_res4_pre_bn(conv3_res3)
        conv3_res4_pre_relu = F.relu(conv3_res4_pre_bn)
        conv3_res4_conv1 = self.conv3_res4_conv1(conv3_res4_pre_relu)
        conv3_res4_conv1_bn = self.conv3_res4_conv1_bn(conv3_res4_conv1)
        conv3_res4_conv1_relu = F.relu(conv3_res4_conv1_bn)
        conv3_res4_conv2_pad = F.pad(conv3_res4_conv1_relu, (1, 1, 1, 1))
        conv3_res4_conv2 = self.conv3_res4_conv2(conv3_res4_conv2_pad)
        conv3_res4_conv2_bn = self.conv3_res4_conv2_bn(conv3_res4_conv2)
        conv3_res4_conv2_relu = F.relu(conv3_res4_conv2_bn)
        conv3_res4_conv3 = self.conv3_res4_conv3(conv3_res4_conv2_relu)
        conv3_res4      = conv3_res3 + conv3_res4_conv3
        conv4_res1_pre_bn = self.conv4_res1_pre_bn(conv3_res4)
        conv4_res1_pre_relu = F.relu(conv4_res1_pre_bn)
        conv4_res1_proj = self.conv4_res1_proj(conv4_res1_pre_relu)
        conv4_res1_conv1_pad = F.pad(conv4_res1_pre_relu, (1, 1, 1, 1))
        conv4_res1_conv1 = self.conv4_res1_conv1(conv4_res1_conv1_pad)
        conv4_res1_conv1_bn = self.conv4_res1_conv1_bn(conv4_res1_conv1)
        conv4_res1_conv1_relu = F.relu(conv4_res1_conv1_bn)
        conv4_res1_conv2 = self.conv4_res1_conv2(conv4_res1_conv1_relu)
        conv4_res1      = conv4_res1_proj + conv4_res1_conv2
        conv4_res2_pre_bn = self.conv4_res2_pre_bn(conv4_res1)
        conv4_res2_pre_relu = F.relu(conv4_res2_pre_bn)
        conv4_res2_conv1_proj = self.conv4_res2_conv1_proj(conv4_res2_pre_relu)
        conv4_res2_conv1 = self.conv4_res2_conv1(conv4_res2_pre_relu)
        conv4_res2_conv1_bn = self.conv4_res2_conv1_bn(conv4_res2_conv1)
        conv4_res2_conv1_relu = F.relu(conv4_res2_conv1_bn)
        conv4_res2_conv2_pad = F.pad(conv4_res2_conv1_relu, (1, 1, 1, 1))
        conv4_res2_conv2 = self.conv4_res2_conv2(conv4_res2_conv2_pad)
        conv4_res2_conv2_bn = self.conv4_res2_conv2_bn(conv4_res2_conv2)
        conv4_res2_conv2_relu = F.relu(conv4_res2_conv2_bn)
        conv4_res2_conv3 = self.conv4_res2_conv3(conv4_res2_conv2_relu)
        conv4_res2      = conv4_res2_conv1_proj + conv4_res2_conv3
        conv4_res3_pre_bn = self.conv4_res3_pre_bn(conv4_res2)
        conv4_res3_pre_relu = F.relu(conv4_res3_pre_bn)
        conv4_res3_conv1 = self.conv4_res3_conv1(conv4_res3_pre_relu)
        conv4_res3_conv1_bn = self.conv4_res3_conv1_bn(conv4_res3_conv1)
        conv4_res3_conv1_relu = F.relu(conv4_res3_conv1_bn)
        conv4_res3_conv2_pad = F.pad(conv4_res3_conv1_relu, (1, 1, 1, 1))
        conv4_res3_conv2 = self.conv4_res3_conv2(conv4_res3_conv2_pad)
        conv4_res3_conv2_bn = self.conv4_res3_conv2_bn(conv4_res3_conv2)
        conv4_res3_conv2_relu = F.relu(conv4_res3_conv2_bn)
        conv4_res3_conv3 = self.conv4_res3_conv3(conv4_res3_conv2_relu)
        conv4_res3      = conv4_res2 + conv4_res3_conv3
        conv5_bn        = self.conv5_bn(conv4_res3)
        conv5_relu      = F.relu(conv5_bn)
        pool5           = F.avg_pool2d(conv5_relu, kernel_size=(4, 4), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        fc1_0           = pool5.view(pool5.size(0), -1)
        fc1_1           = self.fc1_1(fc1_0)
        bn_fc1          = self.bn_fc1(fc1_1)
        #return bn_fc1
        bn_fc1      = bn_fc1.reshape(bn_fc1.size()[0], bn_fc1.size()[1])
        slice_fc1, slice_fc2       = bn_fc1[:, :256], bn_fc1[:, 256:]
        eltwise_fc1 = torch.max(slice_fc1, slice_fc2)

        return eltwise_fc1

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

