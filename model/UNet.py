from model.UNet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        
        self.up1 = up(512, 128, 256)
        self.up2 = up(256, 64, 128)
        self.up3 = up(128, 32, 64)
        self.up4 = up(64, 32, 32)
        
        self.outc = outconv(32,n_classes)
        
        self.weight_init()
        
    def forward(self, x_raw):
        x_raw = torch.squeeze(x_raw,dim=1)

        x1 = self.inc(x_raw)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

def weights_init_kaiming(m):

    class_name = m.__class__.__name__

    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

    elif class_name.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

    elif class_name.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
