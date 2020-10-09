import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchviz import make_dot, make_dot_from_trace

class UNet10(nn.Module):
    """a simple UNet from paper 'Deep Learning for ECG Segmentation'"""
    def __init__(self, in_ch, out_ch):
        super(UNet10, self).__init__()
        # conv1d + batchnorm1d + relu
        self.conv1 = self.ConvNet(in_ch, 4, 9, 1, 4)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = self.ConvNet(4, 8, 9, 1, 4)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = self.ConvNet(8, 16, 9, 1, 4)
        self.pool3 = nn.MaxPool1d(2, stride=2)
        self.conv4 = self.ConvNet(16, 32, 9, 1, 4)
        self.pool4 = nn.MaxPool1d(2, stride=2)
        self.conv5 = self.ConvNet(32, 64, 9, 1, 4)
        self.pool5 = nn.MaxPool1d(2, stride=2) 
        self.conv6 = self.ConvNet(64, 128, 9, 1, 4)
        self.pool6 = nn.MaxPool1d(2, stride=2)
        self.conv7 = self.ConvNet(128, 256, 9, 1, 4)
        self.pool7 = nn.MaxPool1d(2, stride=2)
        self.conv8 = self.ConvNet(256, 512, 9, 1, 4)
        self.pool8 = nn.MaxPool1d(2, stride=2)

        # bottle neck ( conv1d + batchnorm1d + relu)
        self.conv9 = self.ConvNet(512, 1024, 9, 1, 4)

        # upconv1d
        self.upconv1 = self.ConvTransNet(1024)
        self.conv10 = self.ConvNet(1536, 512, 9, 1, 4)
        self.upconv2 = self.ConvTransNet(512)
        self.conv11 = self.ConvNet(768, 256, 9, 1, 4)
        self.upconv3 = self.ConvTransNet(256)
        self.conv12 = self.ConvNet(384, 128, 9, 1, 4)
        self.upconv4 = self.ConvTransNet(128)
        self.conv13 = self.ConvNet(192, 64, 9, 1, 4)
        self.upconv5 = self.ConvTransNet(64)
        self.conv14 = self.ConvNet(96, 32, 9, 1, 4)
        self.upconv6 = self.ConvTransNet(32)
        self.conv15 = self.ConvNet(48, 16, 9, 1, 4)
        self.upconv7 = self.ConvTransNet(16)
        self.conv16 = self.ConvNet(24, 8, 9, 1, 4)
        self.upconv8 = self.ConvTransNet(8)
        self.conv17 = self.ConvNet(12, 4, 9, 1, 4)
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
        mp4 = self.pool4(c4)
        c5 = self.conv5(mp4)
        mp5 = self.pool5(c5)
        c6 = self.conv6(mp5)
        mp6 = self.pool6(c6)
        c7 = self.conv7(mp6)
        mp7 = self.pool7(c7)
        c8 = self.conv8(mp7)
        mp8 = self.pool8(c8)

        c9 = self.conv9(mp8)

        up10 = self.upconv1(c9)
        cat10 = torch.cat((c8, up10), dim=1)
        c10 = self.conv10(cat10)       
        up11 = self.upconv2(c10)
        cat11 = torch.cat((c7, up11), dim=1)
        c11 = self.conv11(cat11)
        up12 = self.upconv3(c11)
        cat12 = torch.cat((c6, up12), dim=1)
        c12 = self.conv12(cat12)
        up13 = self.upconv4(c12)
        cat13 = torch.cat((c5, up13), dim=1)
        c13 = self.conv13(cat13)
        up14 = self.upconv5(c13)
        cat14 = torch.cat((c4, up14), dim=1)
        c14 = self.conv14(cat14)
        up15 = self.upconv6(c14)
        cat15 = torch.cat((c3, up15), dim=1)
        c15 = self.conv15(cat15)
        up16 = self.upconv7(c15)
        cat16 = torch.cat((c2, up16), dim=1)
        c16 = self.conv16(cat16)
        up17 = self.upconv8(c16)
        cat17 = torch.cat((c1, up17), dim=1)
        c17 = self.conv17(cat17)
        f = self.final(c17)
        #f = self.softmax(f)



        return f