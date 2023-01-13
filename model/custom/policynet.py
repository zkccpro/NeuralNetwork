import torchvision.models as models
from torch import nn
import torch
from model import net
from conf import globalParam
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(self, action_num, action_space=None, dropout=0):
        super().__init__()
        self.backbone = models.resnet18().to(globalParam.device)

        self.mean_side = net.FullConnection(1000, action_num, layer_num=2, dropout=dropout)
        self.log_std_side = net.FullConnection(1000, action_num, layer_num=2, dropout=dropout)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, stat):
        epsilon = 1e-6

        x = self.backbone(stat)
        mean = self.mean_side(x)
        log_std = self.log_std_side(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.sample()  # reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    # def sample(self, stat):
    #     epsilon = 1e-6
    #     mean, log_std = self.forward(stat)
    #     std = log_std.exp()
    #     normal = Normal(mean, std)
    #     x_t = normal.sample()  # reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, action_num, action_space=None, dropout=0):
        super().__init__()
        self.backbone = models.resnet18().to(globalParam.device)

        self.mean = net.FullConnection(1000, action_num, layer_num=2, dropout=dropout)
        self.noise = torch.Tensor(action_num).to(globalParam.device)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, stat):
        x = self.backbone(stat)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias

        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean  # action, prob, mean

    # def sample(self, stat):
    #     mean = self.forward(stat)
    #     noise = self.noise.normal_(0., std=0.1)
    #     noise = noise.clamp(-0.25, 0.25)
    #     action = mean + noise
    #     return action, torch.tensor(0.), mean  # action, prob, mean
