import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
import scipy
from mmengine.model import BaseModule, ModuleList
from torchvision.models import resnet18
from mmengine.registry import MODELS
from mmdet.models.backbones import ResNet


def makeGaussian(size, a=1, b=0.25, fwhm1=7, fwhm2=3):
    x1 = np.arange(0, size, 1, float)
    x2 = np.arange(0, size, 1, float)
    y1 = x1[:, np.newaxis]
    y2 = x2[:, np.newaxis]
    x10 = y10 = size // 2
    x20 = y20 = size // 2
    g1 = np.exp(-4 * np.log(2) * ((x1 - x10) ** 2 + (y1 - y10) ** 2) / fwhm1 ** 2)
    g2 = np.exp(-4 * np.log(2) * ((x2 - x20) ** 2 + (y2 - y20) ** 2) / fwhm2 ** 2)
    gaussian = a * g1 - b * g2
    return gaussian


class Conv3Density(BaseModule):
    def __init__(self, in_channels=3, out_channels=3):
        super(Conv3Density, self).__init__()
        BN_MOMENTUM = 0.1
        self.fov_expand_1 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_channels, kernel_size=3, padding=1,
                                      bias=False)
        self.fov_expand_2 = nn.Conv2d(in_channels=8 * out_channels, out_channels=8 * out_channels, kernel_size=3,
                                      padding=1, bias=False)
        self.fov_squeeze_1 = nn.Conv2d(in_channels=8 * out_channels, out_channels=out_channels, kernel_size=3,
                                       padding=1, bias=False)
        # bn
        self.norm1 = nn.BatchNorm2d(8 * out_channels, momentum=BN_MOMENTUM)
        self.norm2 = nn.BatchNorm2d(8 * out_channels, momentum=BN_MOMENTUM)
        self.norm3 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x, reset_grad=True, train_mode=True):
        layer1 = self.act(self.norm1(self.fov_expand_1(x)))
        layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
        layer3 = self.norm3(self.fov_squeeze_1(layer2))
        output = layer3
        return output


class SaliencySampler(BaseModule):
    def __init__(self, saliency_network, task_input_size=1024):
        super(SaliencySampler, self).__init__()
        self.grid_size = 64
        self.padding_size = 30
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = task_input_size
        self.conv_last = nn.Conv2d(2048, 1, kernel_size=1, padding=0, stride=1)
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1))

        # Spatial transformer localization-network
        self.localization = saliency_network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights

        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                            j - self.padding_size) / (self.grid_size - 1.0)

    def create_grid(self, x):
        P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).cuda(),
            requires_grad=False)
        P[0, :, :, :] = self.P_basis
        P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)

        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)

        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter

        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)

        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)

        grid = torch.cat((xgrids, ygrids), 1)

        grid = torch.nn.Upsample(size=(self.input_size_net, self.input_size_net), mode='bilinear')(grid)

        n, c, h, w = grid.size()
        h = torch.linspace(-1.0, 1.0, self.input_size_net).view(-1, 1).repeat(1, self.input_size_net)
        w = torch.linspace(-1.0, 1.0, self.input_size_net).repeat(self.input_size_net, 1)
        grid_o = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid_o = grid_o.repeat(n, 1, 1, 1).type_as(x).to(x.device)

        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        grid = (2*grid_o+8*grid)/10
        grid_reorder = grid.permute(3, 0, 1, 2)

        grid_inv = torch.autograd.Variable(
            torch.rand((2, grid_reorder.shape[1], self.input_size_net, self.input_size_net),
                        device=grid_reorder.device)*2-1)
        grid_inv[:] = float('nan')

        u_cor = (((grid_reorder[0, :, :, :] + 1) / 2) * (self.input_size_net - 1)).int().long().view(
            grid_reorder.shape[1], -1)
        v_cor = (((grid_reorder[1, :, :, :] + 1) / 2) * (self.input_size_net - 1)).int().long().view(
            grid_reorder.shape[1], -1)
        x_cor = torch.arange(0, grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()
        y_cor = torch.arange(0, grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()
        grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1), v_cor, u_cor] = torch.autograd.Variable(x_cor)
        grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1), v_cor, u_cor] = torch.autograd.Variable(y_cor)
        grid_inv[0] = grid_inv[0] / grid_reorder.shape[3] * 2 - 1
        grid_inv[1] = grid_inv[1] / grid_reorder.shape[2] * 2 - 1
        grid_inv = grid_inv.permute(1, 2, 3, 0)
        grid_inv = (2 * grid_o + 8 * grid_inv)/10
        return grid, grid_inv

    def forward(self, x):
        x_list = []
        x_list = self.localization(x)
        xs = nn.ReLU()(x_list[3])
        xs = self.conv_last(xs)
        xs = nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')(xs)
        xs = xs.view(-1, self.grid_size * self.grid_size)
        xs = nn.Softmax()(xs)
        xs = xs.view(-1, 1, self.grid_size, self.grid_size)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid, grid_inv = self.create_grid(xs_hm)
        x_sampled = F.grid_sample(x, grid.float(),mode='nearest', align_corners=True)
        x_unsampled = F.grid_sample(x_sampled, grid_inv.float(), mode='nearest', align_corners=True)

        return x_sampled, x_unsampled, xs


class AlignedModule(BaseModule):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_shallow = nn.Conv2d(128, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        shallow_feature, deep_feature = x
        deep_feature_orign = deep_feature
        h, w = deep_feature.size()[2:]
        size = (h, w)
        shallow_feature = self.down_shallow(shallow_feature)
        shallow_feature = F.interpolate(shallow_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([deep_feature, shallow_feature], 1))
        deep_feature = self.flow_warp(deep_feature_orign, flow, size=size)

        return deep_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


@MODELS.register_module()
class Deformed(BaseModule):
    def __init__(self, deep_planes=None, shallow_planes=None):
        super(Deformed, self).__init__()
        self.deep_planes = deep_planes
        self.shallow_planes = shallow_planes
        self.deep = ResNet(depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
                           norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch')
        self.density_net = Conv3Density()
        self.saliency_sampler = SaliencySampler(self.deep, task_input_size=512)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x_deformed, x_undeformed, xs_hm = self.saliency_sampler(x)
        return x_deformed, x_undeformed, xs_hm
