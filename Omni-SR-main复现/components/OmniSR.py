#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
from ops.OSAG import OSAG
from ops.pixelshuffle import pixelshuffle_block
import torch.nn.functional as F


class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()

        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_out_ch, kernel_size=3, stride=1, padding=1,
                                bias=bias)

        # self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)
        # 使用 Upsample 替代 PixelShuffle
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        # self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

        # 添加多尺度卷积层
        self.conv3x3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv5x5 = nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=2, bias=bias)

        # 允许动态窗口大小的控制
        self.dynamic_window_size = kwargs.get("dynamic_window_size", False)
        # 设置初始的窗口大小
        self.window_size = kwargs.get("window_size", 32)

        # 新增一个卷积层，将 residual 通道数从 64 转换为 3
        self.residual_to_rgb = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # 根据输入尺寸动态计算窗口大小
        if self.dynamic_window_size:
            self.window_size = self.calculate_dynamic_window_size(H, W)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # 使用3x3卷积处理
        out_3x3 = self.conv3x3(out)
        # 使用5x5卷积处理
        out_5x5 = self.conv5x5(out)
        # 融合不同尺度的特征（这里是相加，也可以选择拼接）
        out = out_3x3 + out_5x5  # 你可以选择使用拼接 `torch.cat`

        # 将 residual 通道数转换为 3，确保与输出的通道数一致
        residual_rgb = self.residual_to_rgb(residual)

        # origin
        out = torch.add(self.output(out), residual_rgb)

        # 使用 Upsample 进行上采样
        out = self.upsample(out)
        # # 进行上采样
        # out = self.up(out)

        # 对输出进行裁剪，保证输出尺寸为目标尺寸
        out = out[:, :, :H * self.up_scale, :W * self.up_scale]

        return out

    def calculate_dynamic_window_size(self, H, W):
        # 计算动态窗口大小的方法
        if self.dynamic_window_size:
            # 这里的逻辑可以根据需要进行调整，当前是根据尺寸自适应
            return min(H, W) // 32 * 32  # 窗口大小按照图像尺寸缩放，确保为32的倍数
        else:
            # 如果不需要动态窗口，使用固定窗口大小
            return self.window_size