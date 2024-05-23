import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad):
        """2D Convolution with masked weight for Autoregressive connection"""
        super(MaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1

        mask = torch.ones(ch_out, ch_in, height, width)
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighborhing channels in current pixel
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2 + 1:] = 0
        else:
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


def maskAConv(c_in=3, c_out=256, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""
    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad),
        nn.BatchNorm2d(c_out))


class MaskBConvBlock(nn.Module):
    def __init__(self, h=128, k_size=3, stride=1, pad=1):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""
        super(MaskBConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, h, 1),  # 1x1
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d('B', h, h, k_size, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, 2 * h, 1),  # 1x1
            nn.BatchNorm2d(2 * h)
        )

    def forward(self, x):
        """Residual connection"""
        return self.net(x) + x
    

class PixelCNN(nn.Module):
    def __init__(self, n_channel=3, h=128, discrete_channel=256):
        """PixelCNN Model"""
        super(PixelCNN, self).__init__()

        self.discrete_channel = discrete_channel

        self.MaskAConv = maskAConv(n_channel, 2 * h, k_size=7, stride=1, pad=3)
        MaskBConv = []
        for i in range(15):
            MaskBConv.append(MaskBConvBlock(h, k_size=3, stride=1, pad=1))
        self.MaskBConv = nn.Sequential(*MaskBConv)
        
        self.MaskAConv.to(self.device)
        self.MaskBConv.to(self.device)

        # 1x1 conv to 3x256 channels
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1, stride=1, padding=0))
        self.out.to(self.device)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr= 0.001)

    def forward(self, x):
        """
        Args:
            x: [batch_size, channel, height, width]
        Return:
            out [batch_size, channel, height, width, 256]
        """
        batch_size, c_in, height, width = x.size()

        # [batch_size, 2h, 32, 32]
        x = self.MaskAConv(x)

        # [batch_size, 2h, 32, 32]
        x = self.MaskBConv(x)

        # [batch_size, 3x256, 32, 32]
        x = self.out(x)

        # [batch_size, 3, 256, 32, 32]
        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        # [batch_size, 3, 32, 32, 256]
        x = x.permute(0, 1, 3, 4, 2)

        return x
    
    def testing(self,x):
        image = Variable(x).cuda()
        logit = self.forward(image)
        logit = logit.contiguous()
        logit = logit.view(-1, 256)
        target = Variable(image.data.view(-1) * 255).long()
        batch_loss = self.loss_fn(logit, target)
        return batch_loss
    
    def learning(self, x):
        # [batch_size, 3, 32, 32]
        image = Variable(x).cuda()

        # [batch_size, 3, 32, 32, 256]
        logit = self.forward(image)
        logit = logit.contiguous()
        logit = logit.view(-1, 256)

        target = Variable(image.data.view(-1) * 255).long()
        batch_loss = self.loss_fn(logit, target)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()
        
    def sample(self, x):
        x = x.to(self.device)
        x[:, :, 20:, :] = 0  
        for i in range(20, 32):                                        # generate bottom half
            for j in range(32):
                for k in range(3):
                    out = self.forward(x)                                      # [batch_size, 256, 1, 32, 32]
                    probs = torch.softmax(out[:, :, i, j], dim=2).data  # [batch_size, 256]
                    pixel = torch.multinomial(probs[:,k], 1).float() / 255.
                    x[:, k, i, j] = pixel[:, 0] 
        return x
    """
    def sample(self):
        
        num = 16
        sample = torch.zeros(num, 3, 32, 32).cuda()

        for i in range(32):
            for j in range(32):

                # [batch_size, channel, height, width, 256]
                out = self.forward(Variable(sample, volatile=True))

                # out[:, :, i, j]
                # => [batch_size, channel, 256]
                probs = F.softmax(out[:, :, i, j], dim=2).data

                # Sample single pixel (each channel independently)
                for k in range(3):
                    # 0 ~ 255 => 0 ~ 1
                    pixel = torch.multinomial(probs[:, k], 1).float()
                    for l in range(num):
                        sample[l, k, i, j] = pixel[l][0].to(self.device)
        return sample
    """