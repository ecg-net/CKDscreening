from torch import nn
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F

from constants import INITIAL_KERNEL_NUM, MIN_DROPOUT, MAX_DROPOUT, CONV1_KERNEL1, CONV1_KERNEL2

# +
from torch import nn
from collections import OrderedDict


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,activation,stride=1,padding = 1):
        super(Bottleneck, self).__init__()
        self.stride=stride
        self.conv1 = nn.Conv1d(in_channel,in_channel*expansion,kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channel*expansion,in_channel*expansion,kernel_size = 3, groups = in_channel*expansion,
                               padding=padding,stride = stride)
        self.conv3 = nn.Conv1d(in_channel*expansion,out_channel,kernel_size = 1, stride =1)
        self.b0 = nn.BatchNorm1d(in_channel*expansion)
        self.b1 =  nn.BatchNorm1d(in_channel*expansion)
        self.d = nn.Dropout()
        self.act = activation()
    def forward(self,x):
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x+y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y

from torch import nn
from collections import OrderedDict

class MBConv(nn.Module):
    def __init__(self,in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
        super(MBConv, self).__init__()
        self.stack = OrderedDict()
        for i in range(0,layers-1):
            self.stack['s'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
            #self.stack['a'+str(i)] = activation()
        self.stack['s'+str(layers+1)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
        # self.stack['a'+str(layers+1)] = activation()
        self.stack = nn.Sequential(self.stack)
        
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self,x):
        x = self.stack(x)
        return self.bn(x)


"""def MBConv(in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
    stack = OrderedDict()
    for i in range(0,layers-1):
        stack['b'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
    stack['b'+str(layers)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
    return nn.Sequential(stack)"""


class EffNet(nn.Module):
    
    def __init__(self,num_additional_features = 0,depth = [1,2,2,3,3,3,3],channels = [32,16,24,40,80,112,192,320,1280,1867],
                dilation = 1,stride = 2,expansion = 6,hook=False,multi=False,reg=False, start_channels=12, multi_class=False):
        super(EffNet, self).__init__()
        print("depth ",depth)
        self.stage1 = nn.Conv1d(start_channels, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) # 
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv
        
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, channels[9])
        self.fc.bias.data[0] = 0.275
        self.multi_class= multi_class
        
        self.hook = hook
        self.multi = multi
        self.reg = reg
    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 12 x 2500
        x = self.b0(self.stage1(x))
        # N x 32 x 1250
        x = self.stage2(x)
        # N x 16 x 625
        x = self.stage3(x)
        # N x 24 x 313
        x = self.Pool(x)
        # N x 24 x 313
        
        x = self.stage4(x)
        # N x 40 x 157
        x = self.stage5(x)
        # N x 80 x 79
        x = self.stage6(x)
        # N x 112 x 40
        x = self.Pool(x)
        # N x 192 x 20
        
        x = self.stage7(x)
        # N x 320 x 10
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 10
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        if self.hook:
            return x
        x = self.fc(x)
        # N x 1
        if self.reg:
            return x
        if self.multi_class:
            return torch.softmax(x, dim=1)
        return torch.sigmoid(x) #Use sigmoid for binary classification and softmax for multi-class classification
# -



# # use trial.suggest from optuna to suggest hyperparameters 
# # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


# +
def output_channels(num):
    return int(num / 3) + int(num) + int(num) + int(num / 3)

class Multi_2D_CNN_block(nn.Module):
    def __init__(self, in_channels, num_kernel):
        super(Multi_2D_CNN_block, self).__init__()
        conv_block = BasicConv2d
        self.a = conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1))

        self.b = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 2), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )

        self.c = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 3), int(num_kernel / 2), kernel_size=(3, 3)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, int(num_kernel / 3),kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3),padding=(1,1),stride = 1)
            
        )
        
        self.out_channels = output_channels(num_kernel)
        
        # I get out_channels is total number of out_channels for a/b/c
        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        branch1 = self.a(x)
        branch2 = self.b(x)
        branch3 = self.c(x)
        branch4 = self.d(x)
        output = [branch1, branch2, branch3,branch4]
        final = self.bn(torch.cat(output,
                                 1))  # BatchNorm across the concatenation of output channels from final layer of Branch 1/2/3
        return final
        # ,1 refers to the channel dimension
# -



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class MyModel(nn.Module):

    def __init__(self,trial = None, inital_kernel_num = None,dropout = None,conv1kernel1 = None,conv1kernel2 = None,reduced = False,num_additional_features=0):# trial, inital_kernel_num,dropout,conv1kernel1,conv1kernel2
        super(MyModel, self).__init__()
        multi_2d_cnn = Multi_2D_CNN_block
        conv_block = BasicConv2d
        # Define the values in constants.py and import them
        if not trial is None:
            initial_kernel_num = trial.suggest_categorical("kernel_num", INITIAL_KERNEL_NUM)
            dropout = trial.suggest_float('dropout', MIN_DROPOUT, MAX_DROPOUT)
            conv1kernel1 = trial.suggest_categorical("conv_1_1", CONV1_KERNEL1)
            conv1kernel2 = trial.suggest_categorical("conv_1_2", CONV1_KERNEL2)
        elif not inital_kernel_num is None:
            initial_kernel_num = inital_kernel_num
            dropout = dropout
            conv1kernel1 = conv1kernel1
            conv1kernel2 = conv1kernel2
            self.num_additional_features = num_additional_features
        

        self.conv_1 = conv_block(1, 64, kernel_size=(conv1kernel1, conv1kernel2), stride=(2, 1)) #kernel_size=(7,1), (21,3), (21,1)....
        
        self.multi_2d_cnn_1a = nn.Sequential(
            multi_2d_cnn(in_channels=64, num_kernel=initial_kernel_num),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num), num_kernel=initial_kernel_num),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        
        num_kernel=initial_kernel_num
        self.path_1 = nn.Sequential(
            nn.Conv2d(64, output_channels(initial_kernel_num), kernel_size=(1, 1), stride=(3, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num))
        )
        self.multi_2d_cnn_1b = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num), num_kernel=initial_kernel_num * 1.5),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num*1.5), num_kernel=initial_kernel_num * 1.5),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.path_2 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num), output_channels(initial_kernel_num * 1.5), kernel_size=(1, 1), stride=(3, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 1.5))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 1.5), num_kernel=initial_kernel_num * 2),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 2), num_kernel=initial_kernel_num * 2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.path_3 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num * 1.5), output_channels(initial_kernel_num * 2), kernel_size=(1, 1), stride=(2, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 2))
        )
        
        self.multi_2d_cnn_2a = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 2), num_kernel=initial_kernel_num * 3),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 3), num_kernel=initial_kernel_num * 3),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 3), num_kernel=initial_kernel_num * 4),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.path_4 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num * 2), output_channels(initial_kernel_num * 4), kernel_size=(1, 1), stride=(2, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 4))
        )
        
        self.multi_2d_cnn_2b = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 4), num_kernel=initial_kernel_num * 5),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 5), num_kernel=initial_kernel_num * 6),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 6), num_kernel=initial_kernel_num * 7),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.path_5 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num * 4), output_channels(initial_kernel_num * 7), kernel_size=(1, 1), stride=(2, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 7))
        )
        
        self.multi_2d_cnn_2c = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 7), num_kernel=initial_kernel_num * 8),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 8), num_kernel=initial_kernel_num * 8),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 8), num_kernel=initial_kernel_num * 8),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.path_6 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num * 7), output_channels(initial_kernel_num * 8), kernel_size=(1, 1), stride=(2, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 8))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 8), num_kernel=initial_kernel_num * 12),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 12), num_kernel=initial_kernel_num * 14),
            multi_2d_cnn(in_channels=output_channels(initial_kernel_num * 14), num_kernel=initial_kernel_num * 16),
        )

        self.path_7 = nn.Sequential(
            nn.Conv2d(output_channels(initial_kernel_num * 8), output_channels(initial_kernel_num * 16), kernel_size=(1, 1), stride=(1, 1)),
            
            nn.BatchNorm2d(output_channels(initial_kernel_num * 16))
        )
        neurons = (output_channels(initial_kernel_num * 16))

        self.p1 = nn.AdaptiveAvgPool2d((1, 1))
        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout(dropout)
        self.l1 = nn.Linear(neurons+self.num_additional_features, 1)

    def forward(self, x):
        # N x 1 x 2500 x 12 -> N x 2500 x 12
        if x.shape[2]==2500:
            reduced = True
        else:
            reduced = False
        x = self.conv_1(x)
        # N x 64 x 1250 x 12 tensor -> N x 1250 x
        y = self.path_1(x)[:, :, :-1, :]
        
        x = self.multi_2d_cnn_1a(x)
        x = x+y
        # N x 74 x 416 x 12 tensor
        y = self.path_2(x)[:, :, :-1, :]
        x = self.multi_2d_cnn_1b(x)
        x = x+y
        # N x 112 x 138 x 12 tensor
        if reduced:
            y = self.path_3(x)
        else:
            y = self.path_3(x)[:, :, :-1, :]
        x = self.multi_2d_cnn_1c(x)
        # N x 149 x 69 x 12
        x = x+y
        if reduced:
            y = self.path_4(x)[:, :, :-1, :]
        else:
            y = self.path_4(x)
        x = self.multi_2d_cnn_2a(x)
        x = x+y
        # N x 298 x 69 x 12
        if reduced:
            y = self.path_5(x)
        else:
            y = self.path_5(x)[:, :, :-1, :]
        x = self.multi_2d_cnn_2b(x)
        x = x+y
        # N x 522 x 17 x 12
        if reduced:
            y = self.path_6(x)[:, :, :-1, :]
        else:
            y = self.path_6(x)
        x = self.multi_2d_cnn_2c(x)
        x = x+y
        # N x 597 x 8 x 12
        y = self.path_7(x)
        x = self.multi_2d_cnn_2d(x)
        x = x+y
        # N x 1194 x 8 x 12
        
        # print(x.shape,self.shortcut(x).shape)
        # print(torch.sum(x[0,0,:,:])/(8*12),self.shortcut(x)[0,0,0,0])
        x = self.p1(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.l1(x)
        # x = self.output(x)
        # N x 1
        return x


class Mayo_Net(nn.Module):
    def __init__(self):
        super(Mayo_Net, self).__init__()
        self.conv1 = nn.Conv1d(12,16,5,padding=2)#nn.Conv1d(12,16,5)
        self.batch1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2,1) 
        self.conv2 = nn.Conv1d(16,16,5,padding=2)
        self.conv3 = nn.Conv1d(16,32,5,padding=4)
        self.batch3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(4,1)
        self.conv4 = nn.Conv1d(32,32,3,padding=2)
        self.conv5 = nn.Conv1d(32,64,3,padding=3)
        self.batch5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64,64,3,padding=3)
        self.conv7 = nn.Conv1d(13344,64,3)
        self.conv8 = nn.Conv1d(64,64,3)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(8*64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.missed = 0
        self.total = 0
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = self.batch1(x)
        x = F.relu(self.pool1(x))
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.batch3(x)
        x = F.relu(self.pool1(x))
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(self.pool3(x))
        x = self.conv6(x)
        x = self.batch5(x)
        x = F.relu(self.pool3(x))
        x = x.view(-1, 13344,12)
        x = self.batch5(self.conv7(self.drop(x)))
        x = self.batch5(self.conv8(x))
        x = x.view(-1, 64*8)
        x = self.drop(F.relu(self.batch5(self.fc1(x))))
        x = self.drop(F.relu(self.batch3(self.fc2(x))))
        x = self.fc3(x)
        return x


class Mayo_Net_mortality(nn.Module):
    def __init__(self):
        super(Mayo_Net_mortality, self).__init__()
        self.conv1 = nn.Conv1d(12,16,5,padding=2)#nn.Conv1d(12,16,5)
        self.batch1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2,1) 
        self.conv2 = nn.Conv1d(16,16,5,padding=2)
        self.conv3 = nn.Conv1d(16,32,5,padding=4)
        self.batch3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(4,1)
        self.conv4 = nn.Conv1d(32,32,3,padding=2)
        self.conv5 = nn.Conv1d(32,64,3,padding=3)
        self.batch5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64,64,3,padding=1)
        self.conv7 = nn.Conv1d(26656,64,3)
        self.conv8 = nn.Conv1d(64,64,3)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(8*64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.missed = 0
        self.total = 0
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = self.batch1(x)
        x = F.relu(self.pool1(x))
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.batch3(x)
        x = F.relu(self.pool1(x))
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(self.pool3(x))
        x = self.conv6(x)
        x = self.batch5(x)
        x = F.relu(self.pool3(x))
        x = x.view(-1, 26656,12)
        x = self.batch5(self.conv7(self.drop(x)))
        x = self.batch5(self.conv8(x))
        x = x.view(-1, 64*8)
        x = self.drop(F.relu(self.batch5(self.fc1(x))))
        x = self.drop(F.relu(self.batch3(self.fc2(x))))
        x = self.fc3(x)
        return x
