import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.distance import MultiHeadDistanceLayer
#from torchviz import make_dot, make_dot_from_trace

class UNet6(nn.Module):
    """a simple UNet from paper 'Deep Learning for ECG Segmentation'"""
    def __init__(self, in_ch, out_ch):
        super(UNet6, self).__init__()
        # conv1d + batchnorm1d + relu
        self.conv1 = self.ConvNet(in_ch, 4, 9, 1, 4)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = self.ConvNet(4, 8, 9, 1, 4)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = self.ConvNet(8, 16, 9, 1, 4)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.conv4 = self.ConvNet(16, 32, 9, 1, 4)

        self.distance = MultiHeadDistanceLayer(2, 16, 2560//(2**3), 32)

        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.conv5 = self.ConvNet(32, 64, 9, 1, 4)
        self.pool5 = nn.MaxPool1d(2, stride=2) 
        self.conv6 = self.ConvNet(64, 128, 9, 1, 4)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        # bottle neck ( conv1d + batchnorm1d + relu)
        self.conv7 = self.ConvNet(128, 256, 9, 1, 4)
        # upconv1d
        self.upconv1 = self.ConvTransNet(256)
        self.conv8 = self.ConvNet(384, 128, 9, 1, 4)
        self.upconv2 = self.ConvTransNet(128)
        self.conv9 = self.ConvNet(192, 64, 9, 1, 4)
        self.upconv3 = self.ConvTransNet(64)
        self.conv10 = self.ConvNet(98, 32, 9, 1, 4)
        self.upconv4 = self.ConvTransNet(32)
        self.conv11 = self.ConvNet(48, 16, 9, 1, 4)
        self.upconv5 = self.ConvTransNet(16)
        self.conv12 = self.ConvNet(24, 8, 9, 1, 4)
        self.upconv6 = self.ConvTransNet(8)
        self.conv13 = self.ConvNet(12, 4, 9, 1, 4)
        self.final = nn.Conv1d(4, out_ch, 1)
        self.sigmoid  = nn.Sigmoid ()
        self.softmax = nn.Softmax(1)
        
        # upconv
    def ConvNet(self, in_ch, out_ch, kernel_size, stride, padding):
        net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
                )
        return net

    def ConvTransNet(self, ch):
        net = nn.ConvTranspose1d(ch, ch, 8, stride=2, padding=3) 
        # in_ch should equal to out_ch in this paper
        return net
    
    def forward(self, x):
        c1 = self.conv1(x)
        mp1 = self.pool1(c1)
        c2 = self.conv2(mp1)
        mp2 = self.pool2(c2)
        c3 = self.conv3(mp2)
        mp3 = self.pool3(c3)
        c4 = self.conv4(mp3)

        d4 = self.distance(c4)
        d4 = d4.transpose(1,2)

        mp4 = self.pool4(c4)
        c5 = self.conv5(mp4)
        mp5 = self.pool5(c5)
        c6 = self.conv6(mp5)
        mp6 = self.pool6(c6)
        c7 = self.conv7(mp6)
        up8 = self.upconv1(c7)
        cat8 = torch.cat((c6, up8), dim=1)
        c8 = self.conv8(cat8)
        up9 = self.upconv2(c8)
        cat9 = torch.cat((c5, up9), dim=1)
        c9 = self.conv9(cat9)
        up10 = self.upconv3(c9)

        cat10 = torch.cat((c4,d4,up10), dim=1)

        c10 = self.conv10(cat10)
        up11 = self.upconv4(c10)
        cat11 = torch.cat((c3, up11), dim=1)
        c11 = self.conv11(cat11)
        up12 = self.upconv5(c11)
        cat12 = torch.cat((c2, up12), dim=1)
        c12 = self.conv12(cat12)
        up13 = self.upconv6(c12)
        cat13 = torch.cat((c1, up13), dim=1)
        c13 = self.conv13(cat13)
        f = self.final(c13)
        #f = self.softmax(f)



        return f