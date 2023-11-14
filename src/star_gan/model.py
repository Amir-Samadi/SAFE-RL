import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib 
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.rcsetup as rcsetup

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, agent_type, conv_dim=64, img_channels=3, obs_channels=4, sal_channels=0, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        self.agent_type = agent_type
        self.img_channels = img_channels
        self.obs_channels = obs_channels
        self.sal_channels = sal_channels
        layers.append(nn.Conv2d(img_channels + obs_channels + c_dim + sal_channels, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        
        if self.agent_type in ["SAFE_RL_attGAN"]:
            layers.append(nn.Conv2d(curr_dim, img_channels + 1 + obs_channels + obs_channels + sal_channels, kernel_size=7, stride=1, padding=3, bias=False))
            self.main = nn.Sequential(*layers)     

        elif self.agent_type in ["SAFE_RL_starGAN", "Olson", "GANterfactual"]:
            layers.append(nn.Conv2d(curr_dim, img_channels + obs_channels + sal_channels, kernel_size=7, stride=1, padding=3, bias=False))
            layers.append(nn.Tanh())
            self.main = nn.Sequential(*layers)
            
    def forward(self, x, c, sal=None):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if self.agent_type in ["SAFE_RL_attGAN"]:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            input = torch.cat([x, c, sal], dim=1)
            output = self.main(input)

            content_mask = output[:, :self.img_channels]
            img_attention_mask = F.sigmoid(output[:, self.img_channels:self.img_channels+1])
            obs = output[:, self.img_channels+1:self.img_channels+self.obs_channels+1]
            obs_attention_mask = output[:, self.img_channels+self.obs_channels+1:self.img_channels+self.obs_channels+self.obs_channels+1]
            
            
            img_attention_mask = img_attention_mask.repeat(1, self.img_channels, 1, 1)
            
            img = content_mask * img_attention_mask + x[:, :self.img_channels] * (1 - img_attention_mask)
            # img = x[:, :self.img_channels]
            obs = (obs * obs_attention_mask) + x[:, self.img_channels:self.img_channels+self.obs_channels]*(1-obs_attention_mask)
            sal = F.sigmoid(output[:, self.img_channels+self.obs_channels+self.obs_channels+1:])
            return img,obs,sal
        
        elif self.agent_type in ["SAFE_RL_starGAN"]:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c, sal], dim=1)
            x = self.main(x)
            img = x[:, :self.img_channels]
            obs = x[:, self.img_channels:self.img_channels+self.obs_channels]
            sal = x[:, self.img_channels+self.obs_channels:]
            return img,obs,sal
        elif self.agent_type in ["Olson", "GANterfactual"]:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
            x = self.main(x)
            img = x[:, :self.img_channels]
            obs = x[:, self.img_channels:self.img_channels+self.obs_channels]
            return img,obs,None



class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, img_channels=3, obs_channels=4, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(img_channels + obs_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append(nn.Conv2d(img_channels , conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        
        # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        ##samir
        return out_src, out_cls.mean(dim=(2,3))

