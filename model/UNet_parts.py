#sub_parts of UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """(Conv=>BN=>ReLu)*2"""
    
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
            
    def forward(self, x):
        
        x = self.conv(x)
        return x 

class inconv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()

        self.conv = double_conv(in_channels, out_channels)
        
    def forward(self, x):
    
        x = self.conv(x)
        return x        
        
class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels)
            )
            
    def forward(self, x):
    
        x = self.mpconv(x)
        return x
        
class change_channels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(change_channels, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
        
class up(nn.Module):
    def __init__(self, in_channels, out_channels,\
                 in_ch_x1, mode='sub_pixel', r=2):
        super(up, self).__init__()
        
        if mode == 'bilinear':
            self.up = nn.Upsamples(scale_factor=2, mode='bilinear',\
                                   align_corners=True)
                                   
        elif mode == 'ConvTranspose2d':
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, 2,\
                                         stride=2)
                                         
        elif mode == 'sub_pixel':
            tmp_out_ch = in_ch_x1*r*r
            self.up = nn.Sequential(change_channels(in_ch_x1, tmp_out_ch),\
                                    nn.PixelShuffle(r))
                                    
        self.conv = double_conv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, (diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2))
        
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        
        return x
        
class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()

        self.conv = change_channels(in_channels, out_channels)

    def forward(self, x):
    
        x = self.conv(x)
        return x        
        
        
        
