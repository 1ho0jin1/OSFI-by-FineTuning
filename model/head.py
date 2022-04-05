import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class softmax_head(nn.Module):
    def __init__(self, feat_dim, num_cls):
        super(softmax_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))

    def forward(self, x, label):
        logit = torch.mm(x, self.weight)
        return logit, logit


class normface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32):
        super(normface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)
        logit = self.s * cosine
        return logit, cosine


# CosFace head
class cosface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32, m=0.35):
        super(cosface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logit = self.s * (cosine - one_hot * self.m)

        return logit, cosine



# ArcFace head
class arcface_head(nn.Module):
    def __init__(self,device,feat_dim,num_cls,s,m,
                 easy_margin=True,use_amp=True):
        super(arcface_head,self).__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = torch.Tensor([m]).to(device)
        self.use_amp = use_amp
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x, label):
        cos_m, sin_m = torch.cos(self.m),torch.sin(self.m)
        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cos_theta = torch.mm(x_norm, w_norm).clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.use_amp:
            cos_theta_m = cos_theta_m.to(torch.float16)

        # easy_margin is to stabilize training:
        # i.e., if model is initialized badly s.t. theta + m > pi, then will not use arcface loss!
        if self.easy_margin:
            min_cos_theta = torch.cos(math.pi - self.m)
            cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta)

        idx = torch.zeros_like(cos_theta)
        idx.scatter_(1, label.data.view(-1, 1), 1)
        logit = cos_theta_m * idx + cos_theta * (1-idx)
        logit *= self.s

        return logit, cos_theta



class magface_head(nn.Module):
    def __init__(self,device,feat_dim,num_cls,s,use_amp=True, easy_margin=True,
                 l_a=10, u_a=110, l_m=0.35, u_m=0.7, l_g=40):
        super(magface_head,self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.device = device
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.l_g = l_g
        self.use_amp = use_amp
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def calc_m(self, mag):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (mag - self.l_a) + self.l_m

    def forward(self, x, label=None):
        mag = torch.norm(x, dim=1, keepdim=True)
        mag = mag.clamp(self.l_a, self.u_a)
        m_a = self.calc_m(mag)
        cos_m, sin_m = torch.cos(m_a), torch.sin(m_a)
        g_a = 1 / mag + 1 / (self.u_a ** 2) * mag
        loss_g = self.l_g * g_a.mean()

        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cos_theta = torch.mm(x_norm, w_norm).clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.use_amp:
            cos_theta_m = cos_theta_m.to(torch.float16)

        # easy_margin is to stabilize training:
        # i.e., if model is initialized badly s.t. theta + m > pi, then will not use arcface loss!
        if self.easy_margin:
            min_cos_theta = torch.cos(math.pi - m_a)
            cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta)

        idx = torch.zeros_like(cos_theta)
        idx.scatter_(1, label.data.view(-1, 1), 1)
        logit = cos_theta_m * idx + cos_theta * (1-idx)
        logit *= self.s

        return logit, cos_theta, loss_g
