import torch
from torch import nn
import torchvision
import numpy as np

specs = {"vgg11": ([64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm', ('a', 5)],
                    [4096, 'r', 'd', 4069, 'r', 'd']),
         "vgg16": ([64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm', ('a', 5)],
                    [4096, 'r', 'd', 4069, 'r', 'd']),
         "vgg19": ([64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512, 'm', ('a', 5)],
                    [4096, 'r', 'd', 4069, 'r', 'd']),
         "resnet18": ([('C', 12, 7), 'b', 'r', 'm', ('R', [2, 2, 2, 2]), ('a')],
                    ['r']),
         "resnet34": ([('C', 12, 7), 'b', 'r', 'm', ('R', [3, 4, 6, 3]), ('a')],
                    ['r']),
         "lancet": ([63, 63, 'm', 'd', 32, 32, 'm', 'd', 32, 32, 'm', 'd', 32, 32, 'm', 'd', 64, 64, 'm', 'd', 512, 512, 512, 'm', ('a')],
                    [256, 'r', 'b', 256, 'r', 'b']),
         "lancetwide": ([126, 126, 'm', 'd', 64, 64, 'm', 'd', 64, 64, 'm', 'd', 64, 64, 'm', 'd', 128, 128, 'm', 'd', 512, 512, 512, 'm', ('a')],
                    [256, 'r', 'b', 256, 'r', 'b']),
         "lancet4wide": ([4 * 63, 4 * 63, 'm', 'd', 4 * 32, 4 * 32, 'm', 'd', 4 * 32, 4 * 32, 'm', 'd', 4 * 32, 4 * 32, 'm', 'd', 4 * 64, 4 * 64, 'm', 'd', 512, 512, 512, 'm', ('a')],
                    [256, 'r', 'b', 256, 'r', 'b']),
        "vgg11endbatch": ([64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm', ('a', 5)],
                    [4096, 'r', 'b', 4096, 'r', 'b']),

        }

two_d_specs = {"lancet": ([63, 63, ('m', (1, 4)), 'd', 32, 32, ('m', (1, 4)), 'd', 32, 32, ('m', (1, 4)), 'd', 32, 32, ('m', (2, 4)), 'd', 64, 64, ('m', (2, 4)), 'd', 512, 512, 512, 'm', 'a'],
                    [256, 'r', 'b', 256, 'r', 'b']),
               "lancetwide": ([126, 126, ('m', (1, 4)), 'd', 64, 64, ('m', (1, 4)), 'd', 64, 64, ('m', (1, 4)), 'd', 64, 64, ('m', (2, 4)), 'd', 128, 128, ('m', (2, 4)), 'd', 512, 512, 512, 'm', 'a'],
                    [256, 'r', 'b', 256, 'r', 'b'])
        }

def modify_spec(spec, first_layer_out_channels=None, first_layer_kernel_size=None):
    if first_layer_kernel_size is not None:
        assert first_layer_out_channels is not None
        spec = ([('C', first_layer_out_channels, first_layer_kernel_size)] + spec[0], spec[1])
    return spec

def build_model(modelname_or_spec, num_outputs, num_input_channels, is_2d=False, binary=True, pretrained=False, batch_norm=True, default_conv_kernel=3,
                 drop_prob=0., binary_cutoff=None, first_layer_out_channels=None, first_layer_kernel_size=None):
    if is_2d:
        if first_layer_kernel_size is not None:
            modelname_or_spec = two_d_specs[modelname_or_spec] if type(modelname_or_spec) is str else modelname_or_spec
            modelname_or_spec = modify_spec(modelname_or_spec, first_layer_out_channels, first_layer_kernel_size)
        return ECG2DClassificationModel(modelname_or_spec, num_outputs, num_input_channels, binary, batch_norm, default_conv_kernel, binary_cutoff, drop_prob)
    else:
        if first_layer_kernel_size is not None:
            modelname_or_spec = specs[modelname_or_spec] if type(modelname_or_spec) is str else modelname_or_spec
            modelname_or_spec = modify_spec(modelname_or_spec, first_layer_out_channels, first_layer_kernel_size)
        return ECGCustomClassificationModel(modelname_or_spec, num_outputs, num_input_channels, binary, batch_norm, default_conv_kernel, binary_cutoff, drop_prob)


class ECGModel(torch.nn.Module):
    def __init__(self, num_input_channels, binary=True, binary_cutoff=None):
        super(ECGModel, self).__init__()
        self.num_input_channels = num_input_channels
        self.predictions = None
        self.labels = None
        self.compute_loss = ((lambda yh, y: torch.nn.functional.binary_cross_entropy_with_logits(yh, binary_cutoff <= y)) if binary
                               else torch.nn.functional.mse_loss) 
        if binary:
            assert binary_cutoff is not None

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_step(self, x, y):
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat.flatten(), y.flatten())
        return (y, y_hat, loss)


class ECGCustomClassificationModel(ECGModel):
    def __init__(self, spec, num_outputs, num_input_channels, binary=True, batch_norm=True, default_conv_kernel=3, binary_cutoff=None, drop_prob=0):
        super(ECGCustomClassificationModel, self).__init__(num_input_channels, binary, binary_cutoff)
        self.drop_prob = drop_prob

        self.features, in_channels = self.make_conv(spec, num_input_channels, batch_norm, return_channels=True, default_conv_kernel=default_conv_kernel)
        self.classifier = self.make_fc(spec, in_channels, num_outputs)

    def make_conv(self, spec, in_channels, batch_norm=False, return_channels=False, default_conv_kernel=3):
        # ('C', out_channels, kernel_size=default_conv_kernel, padding=1) # Conv block
        # ('B', out_channels, downsample=False) # ResNet block
        # ('L', out_channels, num_blocks=1) # ResNet Layer (num_blocks blocks, first downsamples)
        # ('R', nbs) # Whole ResNet core
        # ('m', kernel_size=2) # Max pool
        # ('c', out_channels, kernel_size=default_conv_kernel) # Conv
        # ('b') # Batch norm
        # ('r') # ReLU
        # ('a', bins=1) # Adaptive average pooling

        if type(spec) == str:
            spec = specs[spec][0]
        else:
            spec = spec[0]

        layers = []
        for d in spec:
            if type(d) == int:
                d = ('C', d, default_conv_kernel, 1)
            elif type(d) == str:
                d = (d)

            if d[0] == 'C':
                # Conv block
                self.check_length(d, [2, 3, 4])
                if len(d) == 2:
                    d = (d[0], d[1], default_conv_kernel, 1)
                if len(d) == 3:
                    d = (d[0], d[1], d[2], 1)
                conv = nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if batch_norm:
                    layers += [conv, nn.BatchNorm1d(d[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = d[1]

            elif d[0] == 'B':
                # ResNet block
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], False)
                layers += [Block(in_channels, d[1], d[2])]
                in_channels = d[1]

            elif d[0] == 'L':
                # ResNet layer
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], 1)
                layers += self.make_layer(in_channels, d[1], d[2], default_conv_kernel)
                in_channels = d[1]

            elif d[0] == 'R':
                # Whole ResNet core
                self.check_length(d, [2])
                nbs = d[1]
                layers += self.make_layer(in_channels, 64, nbs[0], default_conv_kernel)
                layers += self.make_layer(64, 128, nbs[1], default_conv_kernel)
                layers += self.make_layer(128, 256, nbs[2], default_conv_kernel)
                layers += self.make_layer(256, 512, nbs[3], default_conv_kernel)
                in_channels = 512

            elif d[0] == 'm':
                # max pool
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 2)
                layers += [nn.MaxPool1d(kernel_size=d[1], stride=d[1])] 

            elif d[0] == 'c':
                # single conv
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], default_conv_kernel)
                layers += [nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=1)]
                in_channels = d[1]

            elif d[0] == 'b':
                # batch norm
                if batch_norm:
                    self.check_length(d, [1])
                    layers += [nn.BatchNorm1d(in_channels)]

            elif d[0] == 'r':
                # relu
                self.check_length(d, [1])
                layers += [nn.ReLU(inplace=True)]

            elif d[0] == 'a':
                # Adaptive average pooling
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 1)
                layers += [nn.AdaptiveAvgPool1d(output_size=d[1])]          
                in_channels = d[1] * in_channels

            elif d[0] == 'd':
                self.check_length(d, [1])
                layers += [nn.Dropout(self.drop_prob)]
            else:
                raise NotImplementedError(d)

        if return_channels:
            return nn.Sequential(*layers), in_channels
        else:
            return nn.Sequential(*layers)

    def make_fc(self, spec, in_channels, out_channels):

        if type(spec) == str:
            spec = specs[spec][1]
        else:
            spec = spec[1]

        spec.append(out_channels)

        layers = []
        for d in spec:
            if type(d) == int:
                layers.append(nn.Linear(in_channels, d))
                in_channels = d
            elif d == 'r':
                layers.append(nn.ReLU(inplace=True))
            elif d == 'd':
                layers.append(nn.Dropout(self.drop_prob))
            elif d == 'b':
                layers.append(nn.BatchNorm1d(in_channels))
            else:
                print(d, type(d))
                raise NotImplementedError(d)
        return nn.Sequential(*layers)


    def check_length(self, obj, lengths):
        assert len(obj) in lengths, print('{} is malformed'.format(d))

    def make_layer(self, in_channels, out_channels, n_blocks, kernel_size=3):
        blocks = [Block(in_channels, out_channels, average_pool=True, kernel_size=kernel_size)]
        blocks += [Block(out_channels, out_channels, kernel_size=kernel_size) for _ in range(1, n_blocks)]
        return nn.Sequential(*blocks)


class ECG2DClassificationModel(ECGCustomClassificationModel):
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_conv(self, spec, in_channels, batch_norm=False, return_channels=False, default_conv_kernel=3):
        if type(spec) == str:
            spec = two_d_specs[spec][0]
        else:
            spec = spec[0]

        layers = []
        for d in spec:
            if type(d) == str:
                d = (d,)
            elif type(d) == int:
                d = ('C', int(d), int(default_conv_kernel))
            if d[0] == 'C':
                conv = nn.Conv2d(in_channels, d[1], kernel_size=(d[2], d[2]), padding=(1, 1))
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(d[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = d[1]
            elif d[0] == 'd':
                layers += [nn.Dropout(self.drop_prob)]
            elif d[0] == 'm':
                if len(d) == 1:
                    d = (d[0], (2, 2))
                layers += [nn.MaxPool2d(kernel_size=d[1], stride=d[1])]
            elif d[0] == 'a':
                # Adaptive average pooling
                if len(d) == 1:
                    d = (d[0], (3,3,))
                layers += [nn.AdaptiveAvgPool2d(output_size=d[1])]          
                in_channels = np.prod(d[1]) * in_channels
            else:
                raise NotImplementedError(d)

        if return_channels:
            return nn.Sequential(*layers), in_channels
        else:
            return nn.Sequential(*layers)

    def make_fc(self, spec, in_channels, out_channels):

        if type(spec) == str:
            spec = two_d_specs[spec][1]
        else:
            spec = spec[1]

        spec.append(out_channels)

        layers = []
        for d in spec:
            if type(d) == int:
                layers.append(nn.Linear(in_channels, d))
                in_channels = d
            elif d == 'r':
                layers.append(nn.ReLU(inplace=True))
            elif d == 'd':
                layers.append(nn.Dropout(self.drop_prob))
            elif d == 'b':
                layers.append(nn.BatchNorm1d(in_channels))
            else:
                print(d, type(d))
                raise NotImplementedError(d)
        return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False):
        super(Block, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      padding=1,
                      kernel_size=kernel_size,
                      stride=(2 if average_pool else 1)),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels,
                      out_channels,
                      padding=1,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
            )

        self.downsample = None

        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=2),
                #nn.MaxPool1d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
            x = identity + self.backbone(x)
        else:
            x = identity + self.backbone(x)

        return x


