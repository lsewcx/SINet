import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

def _get_kernel(kernlen=16, nsig=3):
    """
    生成高斯核
    :param kernlen: 核大小
    :param nsig: 标准差
    :return: 高斯核
    """
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def min_max_norm(in_):
    """
    归一化
    :param in_: 输入张量
    :return: 归一化后的张量
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

class SA(nn.Module):
    """
    整体注意力模块
    """
    def __init__(self):
        super(SA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        # 使用高斯核进行卷积操作
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        # 归一化
        soft_attention = min_max_norm(soft_attention)
        # 将输入 x 与 soft_attention 的最大值相乘
        x = torch.mul(x, soft_attention.max(attention))
        return x