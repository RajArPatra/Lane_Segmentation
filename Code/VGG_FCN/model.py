import torch
from torch import nn
from VGG_encoder import *
from dilated_attention_FCN_decoder import *

class LaneVggFCNAttNet(nn.Module):

    def __init__(self):
        super(LaneVggFCNAttNet, self).__init__()

        encode_num_blocks = 5
        in_channels = [3,64,128,256,512]
        out_channels = in_channels[1:]+[512]

        self._encoder = VGGEncoder(encode_num_blocks,in_channels,out_channels)
        decode_layers = ["pool5","pool4","pool3"]
        decode_channels = out_channels[:-len(decode_layers)-1:-1]
        decode_last_stride = 8
        self._decoder = FCNDecoder(decode_layers,decode_channels,decode_last_stride)

        self._score_layer = nn.Conv2d(64,2,1,bias=False)
        self._pix_layer = nn.Sequential(nn.Conv2d(64,3,1,bias=False),nn.ReLU())

    def forward(self,input_tensor):
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)

        decode_logits = self._score_layer(decode_ret)
        binary_seg_pred = torch.argmax(F.softmax(decode_logits,dim=1),dim=1,keepdim=True)
        pix_embedding = self._pix_layer(decode_ret)
        ret = {
            'instance_seg_logits':pix_embedding,
            'binary_seg_pred':binary_seg_pred,
            'binary_seg_logits':decode_logits

        }

        return ret
