import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d

class deform_fusion(nn.Module):
    def __init__(self, opt, in_channels=768*5, cnn_channels=256*3, out_channels=256*3):
        super().__init__()
        #in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if opt.patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2*3*3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1,stride=stride)
        )

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)
        
        return deform_feat

class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768*5+256*3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=self.d_hidn, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,padding=1), 
            nn.ReLU()
        )
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )
    
    def forward(self,f_dis, f_ref, cnn_dis, cnn_ref):
        f_dis = torch.cat((f_dis,cnn_dis),1)
        f_ref = torch.cat((f_ref,cnn_ref),1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f*w).sum(dim=2).sum(dim=2)/w.sum(dim=2).sum(dim=2)

        return pred