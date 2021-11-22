import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
from attention import *

class DilatedAttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DilatedAttentionHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.pa = PositionAttention(inter_channels)
        self.ca = ChannelAttention(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1, bias = False))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1, bias = False))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1, bias = False))

    def forward(self, x):
        feat1 = self.conv5a(x)
        pa_feat = self.pa(feat1)
        pa_conv = self.conv51(pa_feat)
        pa_output = self.conv6(pa_conv)

        feat2 = self.conv5c(x)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv52(ca_feat)
        ca_output = self.conv7(ca_conv)

        feat_sum = pa_conv+ca_conv
        
        paca_output = self.conv8(feat_sum)

        return paca_output
        
class FCNDecoder(nn.Module):

    def __init__(self,decode_layers,decode_channels,decode_last_stride,cout = 64):
        super(FCNDecoder,self).__init__()
        self._in_channels = decode_channels
        self._out_channel = 64
        self._decode_layers = decode_layers
        self.score_net = nn.Sequential()
        self.deconv_net = nn.Sequential()
        self.bn_net = nn.Sequential()
        self.head = DilatedAttentionHead(cout * 8 , cout * 8, nn.BatchNorm2d)
        self.prehead = nn.Sequential(nn.Conv2d(cout, cout * 8, 1, bias = False), nn.BatchNorm2d(cout * 8), nn.ReLU())
        for i,cin in enumerate(self._in_channels):
            self.score_net.add_module("conv"+str(i+1), self._conv_stage(cin,cout))
            if i > 0:
                self.deconv_net.add_module("deconv"+str(i),self._deconv_stage(cout))
        k_size = 2*decode_last_stride
        padding = decode_last_stride//2
        self.deconv_last = nn.ConvTranspose2d(cout * 8 ,cout, k_size, stride = decode_last_stride, padding=padding, bias=False)
    
    def _conv_stage(self,cin,cout):
        return nn.Conv2d(cin,cout,1,stride=1,bias=False)

    def _deconv_stage(self,cout):
        return nn.ConvTranspose2d(cout,cout,4,stride=2,padding = 1,bias=False)
    
        
    def forward(self,encode_data):
        ret = {}
        for i,layer in enumerate(self._decode_layers):
            #print(layer,encode_data[layer].size())
            if i > 0:
                deconv = self.deconv_net[i-1](score)
                #print("deconv from"+self._decode_layers[i-1],deconv.size())
            input_tensor = encode_data[layer]
            score = self.score_net[i](input_tensor)
            #print("conv from"+layer,score.size())
            if i > 0:
                score = deconv + score
        score = self.prehead(score)
        score = self.head(score)
        deconv_final = self.deconv_last(score)
        #print("deconv_final",deconv_final.size())

        return deconv_final
