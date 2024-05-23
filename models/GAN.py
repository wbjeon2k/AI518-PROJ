import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_fid_wrapper as pfw
from torchmetrics.image.fid import FrechetInceptionDistance as FID


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_to_space(x)
        x = self.conv(x)
        return x


class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.space_to_depth = SpaceToDepth(2)

    def forward(self, x):
        x = self.space_to_depth(x)
        x = sum(x.chunk(4, dim=1)) / 4.0
        x = self.conv(x)
        return x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.network = nn.ModuleList([
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1),
            Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
        ])

    def forward(self, x):
        y = x
        for i in range(len(self.network) - 1):
            y = self.network[i](y)
        return self.network[-1](x) + y


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), stride=1, n_filters=256):
        super().__init__()
        self.network = nn.ModuleList([
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size),
            Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
        ])
    def forward(self, x):
        y = x
        for i in range(len(self.network) - 1):
            y = self.network[i](y)
        return self.network[-1](x) + y


class ResBlock(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, padding=1)
        )
    def forward(self, x):
        return self.network(x) + x


class Generator(nn.Module):
    def __init__(self, n_samples = 1024, n_filters = 256):
        super(Generator, self).__init__()
        self.fc = nn.Linear(128, 4 * 4 * 256)
        network = [
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*network)
        self.noise = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z):
        return self.net(self.fc(z).reshape(-1, 256, 4, 4))

    def sample(self, n_samples):
        return self.forward(self.noise.sample([n_samples, 128]).to(self.device))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(128, 4 * 4 * 256)
        network = [
            ResnetBlockDown(3, n_filters=128),
            ResnetBlockDown(128, n_filters=128),
            ResBlock(128, n_filters=128),
            ResBlock(128, n_filters=128),
            nn.ReLU()
        ]
        self.net = nn.Sequential(*network)
        self.fc = nn.Linear(128, 1)

    def forward(self, z):
        return self.fc(torch.sum(self.net(z), dim=(2, 3)))
    
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = Discriminator().to(self.device)
        self.generator = Generator().to(self.device)
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0, 0.9))
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0, 0.9))
    def learning(self, train_loader):
        
        count = 0
        x = train_loader
        if True:
        #for i, x in enumerate(train_loader):
            """
            UserWarning:
            To copy construct from a tensor, 
            it is recommended to use sourceTensor.clone().detach() or 
            sourceTensor.clone().detach().requires_grad_(True),
            rather than torch.tensor(sourceTensor).
            """
            #BEFORE: x = torch.tensor(x).float().to(self.device)
            #FIX:
            x = x.clone().detach().float().to(self.device)
            x = 2 * (x - 0.5)
            self.discriminator_opt.zero_grad()
            generating = self.generator.sample(x.shape[0])
            eps = torch.rand(x.shape[0], 1, 1, 1).to(self.device)
            eps = eps.expand_as(x)
            interpolation = eps * x.data + (1 - eps) * generating.data
            interpolation.requires_grad = True

            discriminator_result = self.discriminator(interpolation)
            grad = torch.autograd.grad(outputs = discriminator_result, inputs = interpolation, grad_outputs = torch.ones(discriminator_result.size()).to(self.device), create_graph = True)[0]
            grad = grad.reshape(128, -1)
            grad_norm = torch.sqrt(torch.sum(grad*grad, dim = 1)+ 1e-12)
            gradient_penalty = ((grad_norm - 1) * (grad_norm - 1)).mean()
            discriminator_loss = self.discriminator(generating).mean() - self.discriminator(x).mean() + 10 * gradient_penalty
            discriminator_loss.backward()
            self.discriminator_opt.step()
            self.generator_opt.zero_grad()
            generating = self.generator.sample(128)
            generator_loss = -self.discriminator(generating).mean()
                #print(generator_loss)
            generator_loss.backward()
            self.generator_opt.step()

            count = count + 1
        return generator_loss
    
    
    def sample(self):
        return ((self.generator.sample(16).permute(0, 2, 3, 1).cpu().detach().numpy())* 0.5 + 0.5) * 255
    
    def testing(self, x : torch.Tensor):
        #pytorch_fid_wrapper/fid_score.py", line 166, in calculate_frechet_distance
        #https://github.com/bioinf-jku/TTUR/issues/4
        #https://github.com/bioinf-jku/TTUR/issues/4#issuecomment-363432195
        #number of samples should be large enough to prevent numerical error
        x = x.clone().detach().float().to(self.device)
        x = 2 * (x - 0.5)
        fake = self.generator.sample(x.shape[0])
        x_int = x.clone().detach().to(dtype=torch.uint8).to(self.device)
        # set pfw device or fid will be done in cpu;;
        # return pfw.fid(torch.Tensor(fake), torch.Tensor(x), device = self.device)
        
        # replace pfw with torchmetrics FID to test if numerical error is solved
        fid  = FID(feature=2048, input_img_size=(x.shape[1], x.shape[2], x.shape[3])).to(self.device)
        fid.update(x_int, real=True)
        fid.update(fake, real=False)
        return fid.compute()