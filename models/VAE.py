import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim = 16):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1), # 16 x 16
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1), # 8 x 8
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 2, 1), # 4 x 4
        nn.ReLU(),
        nn.Flatten(), # 16
        nn.Linear(4 * 4 * 256, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 4 * 4 * 128),
        nn.ReLU(),
        nn.Unflatten(1, (128,4,4)),
        nn.ConvTranspose2d(128, 128, 4, 2, 1), # 8 x 8
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), # 16 x 16
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), # 32 x 32
        nn.ReLU(),
        nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(self.encoder, self.decoder)
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.001)

    def sample(self, z):
        z = torch.randn(100, 16)
        samples = torch.clamp(self.decoder(z), -1, 1)
        samples = samples.permute(0, 2, 3, 1).detach().numpy() * 0.5 + 0.5
        samples = samples * 255
        return samples

    def loss(self, x):
        x = x.to(self.device)
        x = 2 * x - 1
        mu, sigma = self.encoder(x).chunk(2, dim = 1)
        z = torch.randn_like(mu) * sigma.exp() + mu
        reconstruction = self.decoder(z)

        reconstruction_loss = F.mse_loss(x,reconstruction, reduction='none').view(x.shape[0], -1).sum(1).mean()

        kl_loss = (-sigma - 0.5) + (torch.exp(2 * sigma) + mu ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()
        total_loss = reconstruction_loss+kl_loss
        return total_loss
    
    def reconstrucion(self, x):
        x = 2 * x - 1
        z, _ = self.encoder(x).chunk(2, dim = 1)
        x_recon = torch.clamp(self.decoder(z), -1, 1)
        reconstructions = torch.stack((x, x_recon), dim=1).view(-1, 3, 32, 32) * 0.5 + 0.5
        reconstructions = reconstructions.permute(0, 2, 3, 1).detach().numpy() * 255

        return reconstructions
        
    def learning(self, x):
        train_loss = self.loss(x)
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()       

    def testing(self, x):
        return self.loss(x)
