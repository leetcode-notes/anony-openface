"""
This module contains the setup for the models for the 4 main experiment
"""
# from comet_ml import Experiment as comex
import torch
import torch.nn as nn
import torch.nn.functional as f
import math


def make_layers(cfg, in_channels=3, batch_norm=False):
    """
    This function creates the convolutional layers
    :param cfg: dictionary containing details of different convolution layer configurations
    :param in_channels: number of input channels
    :param batch_norm: whether batch normalization is necessary
    :return: setup of convolution layers
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DriverVGG4(nn.Module):
    """
    VGG for Experiment 4. Fine-tuned on EmotioNet for multi-label classification of AUs

    ..Input shape: (B x N x H x W)

        where B is batch_size
              N is number of channels, by default set to 3
              H is image height, by default set to 224
              W is image width, by default set to 224

    ..Output shape: (B x 11)

        where 11 is a probability distribution over 11 classes giving the probabilities of
        occurrence of each AU
    """

    def __init__(self, features, num_classes=11, requires_grad=True):
        """

        :param features: model setup details
        :param num_classes: number of Action Unit classes
        :param requires_grad: specify if autograd will back prop on this network
        """
        super(DriverVGG4, self).__init__()
        self.features = make_layers(features)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if requires_grad is False:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = f.sigmoid(x)
        # x = f.log_softmax(x)
        return x


class CorrCNN(nn.Module):
    """
    Fully Connected network for predicting correlation factor between 2 Action Units

    ..Input shape: Tensor(B x 150528)

    ..Output shape: Tensor(B x 1)

    :param: size: flattened 1D weighted tensor containing image, AU_i and AU_j
                    [[...img...], w_i*AU_i, w_j*AU_j]
    """

    def __init__(self):
        super(CorrCNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)) # because we have 5 AUs
        self.in_layer = nn.Linear(in_features=401408, out_features=11 * 5)
        # self.hidden_layer1 = nn.Linear(in_features=4096, out_features=32)
        # self.out_layer = nn.Linear(in_features=32, out_features=(11 * 5))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.in_layer(x)
        # x = self.hidden_layer1(x)
        # x = self.out_layer(x)
        x = f.tanh(x)
        return x
