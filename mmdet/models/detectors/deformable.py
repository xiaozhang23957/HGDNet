
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image
from mmdet.models.builder import DEFORM, build_loss
from mmengine.visualization import Visualizer
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
import scipy

foo = SummaryWriter(comment='test')


class Conv3DensityGroup(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.1):
        super(Conv3DensityGroup, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3)
        self.GroupNorm1 = nn.GroupNorm(8, 16)
        self.GroupNorm2 = nn.GroupNorm(8, 32)
        self.GroupNorm3 = nn.GroupNorm(8, 16)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.act(self.GroupNorm1(x))
        x = self.conv2(x)
        x = self.act(self.GroupNorm2(x))
        x = self.conv3(x)
        x = self.act(self.GroupNorm3(x))
        x = self.conv4(x)
        return xs


class DeformSampler(nn.Module):
    def __init__(self, grid_size=256, padding_size=32, deform_input_size=512):
        super(DeformSampler, self).__init__()
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = deform_input_size
        self.gaussian_weights = torch.FloatTensor(self.makeGaussian(2 * self.padding_size + 1))
        # Spatial transformer localization-network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        with torch.no_grad():
            self.filter.weight[0].data[:, :, :] = self.gaussian_weights
        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                            j - self.padding_size) / (self.grid_size - 1.0)
        self.P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).cuda(),
            requires_grad=False)
        self.P[0, :, :, :] = self.P_basis
        self.h = torch.linspace(-1.0, 1.0, self.input_size_net).view(-1, 1).repeat(1, self.input_size_net)
        self.w = torch.linspace(-1.0, 1.0, self.input_size_net).repeat(self.input_size_net, 1)
        self.grid_o = torch.cat((self.w.unsqueeze(2), self.h.unsqueeze(2)), 2)

    def makeGaussian(self, size, a=1, b=0.2, fwhm1=100, fwhm2=20):
        """Create a Gaussian kernel."""
        x1, y1 = np.meshgrid(np.arange(size), np.arange(size))
        center = size // 2
        g1 = np.exp(-4 * np.log(2) * ((x1 - center) ** 2 + (y1 - center) ** 2) / fwhm1 ** 2)
        g2 = np.exp(-4 * np.log(2) * ((x1 - center) ** 2 + (y1 - center) ** 2) / fwhm2 ** 2)
        return a * g1 - b * g2

    def create_grid(self, x, p, n, h, w):
        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(p, x_cat).view(-1, 1, self.global_size, self.global_size)
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
        grid = nn.Upsample(size=(h, w), mode='bilinear')(grid)
        grid_o = self.grid_o.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        if not (self.input_size_net == h and self.input_size_net == w):
            grid_o = grid_o.permute(0, 3, 1, 2)
            grid_o = nn.Upsample(size=(h, w), mode='bilinear')(grid_o)
            grid_o = grid_o.permute(0, 2, 3, 1)
        grid_l = grid.permute(0, 2, 3, 1)
        # grid_r = (98 * grid_o + 2 * grid_l) / 100
        # better but not stable
        grid_r = (99 * grid_o + 1 * grid_l) / 100
        # stable but not better
        return grid_r, grid

    def forward(self, x, xs):
        if not self.grid_size == xs.size()[2]:
            xs = nn.Upsample(size=(self.grid_size, self.grid_size), mode='bilinear')(xs)
        n, c, h, w = x.size()
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid_s, grid = self.create_grid(xs_hm, self.P, n, int(h/2), int(w/2))
        x_sampled = F.grid_sample(x, grid_s, align_corners=True)

        return x_sampled, grid, grid_s

class RegistrationModulev2(nn.Module):

    def __init__(self, size, out_channels):
        super(RegistrationModulev2, self).__init__()
        self.size = size
        self.flow_conv1 = nn.Conv2d(out_channels * 2, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.flow_conv2 = nn.Conv2d(out_channels // 4, 2, kernel_size=3, padding=1, bias=False)
        self.GroupNorm1 = nn.GroupNorm(32, out_channels // 4)
        self.GroupNorm2 = nn.GroupNorm(2, 2)
        self.Relu = nn.ReLU(inplace=True)

        self.norm = torch.tensor([[[[self.size, self.size]]]]).cuda()
        self.h = torch.linspace(-1.0, 1.0, self.size).view(-1, 1).repeat(1, self.size)
        self.w = torch.linspace(-1.0, 1.0, self.size).repeat(self.size, 1)
        self.grid = torch.cat((self.w.unsqueeze(2), self.h.unsqueeze(2)), 2)

    def forward(self, x_u, x_d):
        H, W = x_d.shape[2:]
        x_u_inter = F.interpolate(x_u, size=(H, W), mode='bilinear', align_corners=False)
        flow = self.Relu(self.GroupNorm1(self.flow_conv1(torch.cat([x_u_inter, x_d], 1))))
        flow = self.Relu(self.GroupNorm2(self.flow_conv2(flow)))
        x_a, grid_f, grid = self.flow_warp(x_d, flow)
        return x_a, x_u_inter, grid_f, grid

    def flow_warp(self, input, flow):
        n, c, h, w = flow.size()
        grid = self.grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        if not (h == self.size and w == self.size):
            grid = grid.permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)
            grid = grid.permute(0, 2, 3, 1)
        flow = flow.permute(0, 2, 3, 1)
        flow = flow / self.norm
        grid_f = grid + flow
        output = F.grid_sample(input, grid_f, align_corners=True)
        return output, grid_f, grid


class FRIFusion(nn.Module):
    def __init__(self, init_channels, r=4):
        super(FRIFusion, self).__init__()
        channels = init_channels
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_a, x_u):
        x = x_a + x_u
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        weight = self.sigmoid(xlg)
        xo = x_a * weight + x_u * (1 - weight)
        return xo


class Smooth(nn.Module):
    """
    Smooth gradient loss
    """

    def __init__(self, penalty='l1'):
        super(Smooth, self).__init__()
        self.penalty = penalty

    def _diffs(self, y):
        ndims = y.ndimension() - 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            if i == 0:
                y = y.permute(d, 0, 1, 3)
                dfi = y[1:, ...] - y[:-1, ...]
                df[i] = dfi.permute(1, d, 0, 3)
            elif i == 1:
                y = y.permute(d, 0, 1, 2)
                dfi = y[1:, ...] - y[:-1, ...]
                df[i] = dfi.permute(1, 2, d, 0)

        return df

    def forward(self, pred):
        ndims = pred.ndimension() - 2
        if pred.is_cuda:
            df = torch.zeros(1).cuda()
            df.requires_grad_(True)
        else:
            df = torch.zeros(1)
            df.requires_grad_(True)
        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df.data += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df.data += f.pow(2).mean() / ndims
        return df


class Re(nn.Module):
    """
    Registration loss
    """

    def __init__(self,):
        super(Re, self).__init__()

    def forward(self, grid_f, grid_s, grid):
        n, h, w, c = grid_f.size()

        grid_s = grid_s.permute(0,3,1,2)
        grid_s = F.interpolate(grid_s, size=(h,w), mode='bilinear', align_corners=False)
        grid_s = grid_s.permute(0,2,3,1)
        loss_re = F.l1_loss(grid_f+grid_s, 2 * grid)
        return loss_re


@DEFORM.register_module()
class Deform(nn.Module):
    def __init__(self,
                 density_use_layer=0,
                 density_channels=None,
                 loss_mask=None,
                 loss_re=None,
                 grid_size=256,
                 padding_size=15,
                 deform_input_size=512,
                 sigma=25,
                 depth=50,
                 pretrained=None,
                 pretrained_deform=None):
        super(Deform, self).__init__()
        self.density_use_layer = density_use_layer
        self.density_channels = density_channels
        self.loss_mask = build_loss(loss_mask)
        self.grid_size = grid_size
        self.padding_size = padding_size
        self.deform_input_size = deform_input_size
        self.sigma = sigma
        density_channel = self.density_channels[self.density_use_layer]
        self.density_net = Conv3DensityGroup()
        self.loss_grad = Smooth()
        # self.loss_re = Re()
        from mmdet.models.backbones import ResNet
        self.deform_branch = ResNet(depth=depth, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
                                    norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True)
        self.deform_sampler = DeformSampler(self.grid_size, self.padding_size, self.deform_input_size)
        self.registration = RegistrationModulev2(int(deform_input_size / 4), density_channel)
        self.FRIFusion = FRIFusion(density_channel)
        self.init_weights(pretrained=pretrained, pretrained_res=pretrained_deform)

    def init_weights(self, pretrained=None, pretrained_res=None):

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')
        if isinstance(pretrained_res, str):
            logger = logging.getLogger()
            load_checkpoint(self.deform_branch, pretrained_res, strict=False, logger=logger)
        elif pretrained_res is None:
            # use default initializer or customized initializer in subclasses
            pass

    def get_gt_mask_gaussian(self, x, img, gt_obboxes):
        B, C, H, W = x.shape
        density_map = torch.zeros(B, 1, H, W).to(x.device)
        kernel = self.gaussian_kernel(4)
        h, w = img.shape[2:]
        for b in range(B):
            N = gt_obboxes[b].shape[0]
            for n in range(N):
                box = gt_obboxes[b][n]
                cx = box[0]
                cy = box[1]
                l = box[2]
                s = box[3]
                cx = cx * W / w
                cy = cy * H / h
                point_x = cx.long()
                point_y = cy.long()
                full_kernel_size = kernel.shape[0]
                kernel_size = full_kernel_size // 2

                min_img_x = max(0, point_x - kernel_size)
                min_img_y = max(0, point_y - kernel_size)
                max_img_x = min(point_x + kernel_size + 1, W - 1)
                max_img_y = min(point_y + kernel_size + 1, H - 1)
                assert max_img_x > min_img_x
                assert max_img_y > min_img_y

                kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
                kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
                kernel_x_max = kernel_x_min + max_img_x - min_img_x
                kernel_y_max = kernel_y_min + max_img_y - min_img_y
                assert kernel_x_max > kernel_x_min
                assert kernel_y_max > kernel_y_min

                density_map[b, 0, min_img_y:max_img_y, min_img_x:max_img_x] += kernel[kernel_y_min:kernel_y_max,
                                                                               kernel_x_min:kernel_x_max]

        return density_map

    def get_gt_mask_gaussian_finer(self, x, img, gt_obboxes):
        B, C, H, W = x.shape
        density_map = torch.zeros(B, 1, H, W).to(x.device)
        kernel_s = self.gaussian_kernel(2) #41*41
        kernel_m = self.gaussian_kernel(4) #81*81
        kernel_l = self.gaussian_kernel(6) #121*121
        h, w = img.shape[2:]
        for b in range(B):
            N = gt_obboxes[b].shape[0]
            for n in range(N):
                box = gt_obboxes[b][n]
                cx = box[0]
                cy = box[1]
                ch = box[2]
                cw = box[3]
                cx = cx * W / w
                cy = cy * H / h
                point_x = cx.long()
                point_y = cy.long()
                if ch<32 and cw<32:
                    kernel = kernel_s
                elif ch<96 and cw<96:
                    kernel = kernel_m
                else:
                    kernel = kernel_l
                full_kernel_size = kernel.shape[0]
                kernel_size = full_kernel_size // 2

                min_img_x = max(0, point_x - kernel_size)
                min_img_y = max(0, point_y - kernel_size)
                max_img_x = min(point_x + kernel_size + 1, W - 1)
                max_img_y = min(point_y + kernel_size + 1, H - 1)
                assert max_img_x > min_img_x
                assert max_img_y > min_img_y

                kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
                kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
                kernel_x_max = kernel_x_min + max_img_x - min_img_x
                kernel_y_max = kernel_y_min + max_img_y - min_img_y
                assert kernel_x_max > kernel_x_min
                assert kernel_y_max > kernel_y_min

                density_map[b, 0, min_img_y:max_img_y, min_img_x:max_img_x] += kernel[kernel_y_min:kernel_y_max,
                                                                               kernel_x_min:kernel_x_max]

        return density_map

    def gaussian_kernel(self, size):
        sigma = self.sigma
        kernel_size = sigma * size # 4
        img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)
        arr = np.zeros(img_shape)
        arr[img_center] = 1
        arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.max()
        kernel = torch.from_numpy(kernel).cuda()
        return kernel

    def forward_train(self, x_uni, img, gt_obboxes):
        use_layer = self.density_use_layer
        x_uni = x_uni[use_layer]
        xs = self.density_net(img)
        target_label = self.get_gt_mask_gaussian_finer(xs, img, gt_obboxes)
        loss_mask = self.loss_mask(xs, target_label)
        x_deformed, grid, grid_s = self.deform_sampler(img, xs)
        loss_smooth = self.loss_grad(grid)
        x_deformed = self.deform_branch.conv1(x_deformed)
        x_deformed = self.deform_branch.bn1(x_deformed)
        x_deformed = self.deform_branch.relu(x_deformed)
        x_deformed = self.deform_branch.maxpool(x_deformed)
        outs = []
        for i, layer_name in enumerate(self.deform_branch.res_layers):
            res_layer = getattr(self.deform_branch, layer_name)
            if i == 0:
                x_deformed = res_layer(x_deformed)
                x_align, x_u, grid_f, grid = self.registration(x_uni, x_deformed)
                x = self.FRIFusion(x_align, x_u)
            else:
                x = res_layer(x)
            outs.append(x)
        outs[0] = x_uni
        return loss_mask, loss_smooth, tuple(outs)

    def simple_test(self, x_uni, img):
        use_layer = self.density_use_layer
        x_uni = x_uni[use_layer]
        xs = self.density_net(img)
        x_deformed, _,_ = self.deform_sampler(img, xs)

        x_deformed = self.deform_branch.conv1(x_deformed)
        x_deformed = self.deform_branch.bn1(x_deformed)
        x_deformed = self.deform_branch.relu(x_deformed)
        x_deformed = self.deform_branch.maxpool(x_deformed)

        outs = []
        for i, layer_name in enumerate(self.deform_branch.res_layers):
            res_layer = getattr(self.deform_branch, layer_name)
            if i == 0:
                x_deformed = res_layer(x_deformed)
                x_align, x_u, _, _ = self.registration(x_uni, x_deformed)
                # x=x_u
                x = self.FRIFusion(x_align, x_u)
                #
                visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
                drawn_img0 = visualizer.draw_featmap(x_deformed[:,163,:,:], channel_reduction='select_max')
                visualizer.add_image('x_deformed', drawn_img0)
                drawn_img1 = visualizer.draw_featmap(x_uni[:,163,:,:], channel_reduction='select_max')
                visualizer.add_image('x_u', drawn_img1)
                drawn_img2 = visualizer.draw_featmap(x_align[:,163,:,:], channel_reduction='select_max')
                visualizer.add_image('x_align', drawn_img2)
                drawn_img3 = visualizer.draw_featmap(x[:,163,:,:], channel_reduction='select_max')
                visualizer.add_image('x_fusion', drawn_img3)
            else:
                x = res_layer(x)
            outs.append(x)
        outs[0] = x_uni
        return tuple(outs)
