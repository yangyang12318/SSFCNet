from torch.autograd import Variable
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import torch



class BoundaryLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth

    def forward(self, logits, targets):
        # 计算Dice损失

        intersection = (logits * targets).sum()
        union = logits.sum() + targets.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union

        # 计算二进制交叉熵损失
        bce_loss =  nn.BCELoss()(logits, targets)

        # 组合两种损失
        loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss

        return loss


class BoundaryLoss1(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, smooth=1e-5):
        super(BoundaryLoss1, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth

    def forward(self, logits, targets):
        # 计算Dice损失
        siglogits=torch.sigmoid(logits)
        intersection = (siglogits * targets).sum()
        union = siglogits.sum() + targets.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union

        # 计算二进制交叉熵损失
        bce_loss =  nn.BCEWithLogitsLoss()(logits, targets)

        # 组合两种损失
        loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss

        return loss

class BoundaryLoss2(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5,weight_bdy=0.5, smooth=1e-5):
        super(BoundaryLoss2, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_bdy = weight_bdy
        self.smooth = smooth

    def forward(self, logits, targets):
        # 计算Dice损失
        siglogits=torch.sigmoid(logits)
        intersection = (siglogits * targets).sum()
        union = siglogits.sum() + targets.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union

        # 计算二进制交叉熵损失
        bce_loss =  nn.BCEWithLogitsLoss()(logits, targets)

        #边界损失
        probs = torch.cat([1.0 - siglogits, siglogits], dim=1)
        # 2) 生成 one-hot 标签与距离图（批量版）
        with torch.no_grad():
            target_oh = class2one_hot(targets, C=2)  # (B,2,H,W)
            dist_maps = one_hot2dist_batch(target_oh)  # (B,2,H,W)

        # 3) 计算 SurfaceLoss（默认只计前景通道 idc=[1]）
        criterion = SurfaceLoss()
        loss_boundary = criterion(probs, dist_maps, targets)
        # 组合两种损失
        loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss +self.weight_bdy*loss_boundary

        return loss

class Dice(nn.Module):
    def __init__(self,  smooth=1e-5):
        super(Dice, self).__init__()

        self.smooth = smooth

    def forward(self, logits, targets):
        # 计算Dice损失
        siglogits=torch.sigmoid(logits)
        intersection = (siglogits * targets).sum()
        union = siglogits.sum() + targets.sum() + self.smooth
        dice_loss = 1 - (2 * intersection + self.smooth) / union



        return dice_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class torch_MS_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(torch_MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_loss = _ssim(img1, img2, window, self.window_size, channel, self.size_average)

        bce_loss = nn.BCELoss()(img1, img2)

        loss = bce_loss + 1 - ssim_loss

        return loss / 2



import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    """
    Dice + BCE（边界加权） + Sobel梯度一致性（只在GT边界环上计算）
    logits : [B,1,H,W]  未过 sigmoid
    targets: [B,1,H,W]  二值 {0,1}

    关键点：
    - ring_width 决定“边界环”宽度（像素）
    - ring_alpha 放大环上像素的权重，越大越“收边”
    - weight_grad 控制梯度一致性项权重
    - weight_ori 控制方向一致性（余弦）的子权重
    """
    def __init__(
        self,
        weight_dice: float = 0.5,
        weight_bce: float = 0.5,
        ring_width: int = 2,
        ring_alpha: float = 6.0,
        weight_grad: float = 0.2,
        weight_ori: float  = 0.5,  # 方向一致性在梯度项中的占比（0~1）
        eps: float = 1e-6,
        return_components: bool = False,  # 需要时返回各子损失
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce  = weight_bce
        self.ring_width  = ring_width
        self.ring_alpha  = ring_alpha
        self.weight_grad = weight_grad
        self.weight_ori  = weight_ori
        self.eps = eps
        self.return_components = return_components

        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')

        # Sobel 核
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    # ---------- 辅助函数 ----------
    @staticmethod
    def _ensure_shape(x, like):
        # 保证 targets 形状是 [B,1,H,W]
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)
        if x.shape[2:] != like.shape[2:]:
            raise ValueError(f"size mismatch: {x.shape} vs {like.shape}")
        return x

    @staticmethod
    def _morph_grad(x, width):
        """ 近似形态学梯度：dilate - erode（用 max_pool 实现），再膨胀一圈保证连通 """
        dil = F.max_pool2d(x, kernel_size=2*width+1, stride=1, padding=width)
        ero = -F.max_pool2d(-x, kernel_size=2*width+1, stride=1, padding=width)
        edge = (dil - ero).clamp(0, 1)
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        return edge

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + self.eps)
        return gx, gy, mag

    def _weighted_bce(self, logits, targets, weight):
        per_pix = self.bce_logits(logits, targets)  # [B,1,H,W]
        return (per_pix * weight).mean()

    def _weighted_dice(self, probs, targets, weight):
        # 2*sum(w*p*t)/(sum(w*p)+sum(w*t))
        wp = (weight * probs).sum()
        wt = (weight * targets).sum()
        winter = (weight * probs * targets).sum()
        return 1.0 - (2.0 * winter + self.eps) / (wp + wt + self.eps)

    def _grad_loss_on_ring(self, probs, targets, ring):
        # 只在 ring 上计算幅值与方向一致性，避免被背景稀释
        gx_p, gy_p, mag_p = self._sobel(probs)
        gx_t, gy_t, mag_t = self._sobel(targets)

        # 避免 ring 为空
        denom = ring.sum().clamp_min(1.0)

        # 1) 幅值一致（L1）
        mag_l1 = ((mag_p - mag_t).abs() * ring).sum() / denom

        # 2) 方向一致（1 - cosine）
        cos = (gx_p*gx_t + gy_p*gy_t) / (mag_p*mag_t + self.eps)
        ori_loss = ((1.0 - cos) * ring).sum() / denom

        return (1.0 - self.weight_ori) * mag_l1 + self.weight_ori * ori_loss

    # ---------- 前向 ----------
    def forward(self, logits, targets):
        # 预处理
        targets = self._ensure_shape(targets, logits).float()
        probs   = torch.sigmoid(logits)

        # 边界权重图
        with torch.no_grad():
            ring = self._morph_grad(targets, self.ring_width)  # {0,1}
            weight = 1.0 + self.ring_alpha * ring

        # 子损失
        bce_loss  = self._weighted_bce(logits, targets, weight)
        dice_loss = self._weighted_dice(probs, targets, weight)
        grad_loss = self._grad_loss_on_ring(probs, targets, ring)

        # 组合
        total = self.weight_dice * dice_loss + self.weight_bce * bce_loss \
                + self.weight_grad * grad_loss

        if self.return_components:
            return {
                "loss": total,
                "dice": dice_loss.detach(),
                "bce":  bce_loss.detach(),
                "grad": grad_loss.detach()
            }
        return total


class BoundaryAwareLoss_no_bce(nn.Module):
    """
    Dice + BCE（边界加权） + Sobel梯度一致性（只在GT边界环上计算）
    logits : [B,1,H,W]  未过 sigmoid
    targets: [B,1,H,W]  二值 {0,1}

    关键点：
    - ring_width 决定“边界环”宽度（像素）
    - ring_alpha 放大环上像素的权重，越大越“收边”
    - weight_grad 控制梯度一致性项权重
    - weight_ori 控制方向一致性（余弦）的子权重
    """
    def __init__(
        self,
        weight_dice: float = 0.5,

        ring_width: int = 2,
        ring_alpha: float = 6.0,
        weight_grad: float = 0.2,
        weight_ori: float  = 0.5,  # 方向一致性在梯度项中的占比（0~1）
        eps: float = 1e-6,
        return_components: bool = False,  # 需要时返回各子损失
    ):
        super().__init__()
        self.weight_dice = weight_dice

        self.ring_width  = ring_width
        self.ring_alpha  = ring_alpha
        self.weight_grad = weight_grad
        self.weight_ori  = weight_ori
        self.eps = eps
        self.return_components = return_components



        # Sobel 核
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    # ---------- 辅助函数 ----------
    @staticmethod
    def _ensure_shape(x, like):
        # 保证 targets 形状是 [B,1,H,W]
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)
        if x.shape[2:] != like.shape[2:]:
            raise ValueError(f"size mismatch: {x.shape} vs {like.shape}")
        return x

    @staticmethod
    def _morph_grad(x, width):
        """ 近似形态学梯度：dilate - erode（用 max_pool 实现），再膨胀一圈保证连通 """
        dil = F.max_pool2d(x, kernel_size=2*width+1, stride=1, padding=width)
        ero = -F.max_pool2d(-x, kernel_size=2*width+1, stride=1, padding=width)
        edge = (dil - ero).clamp(0, 1)
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        return edge

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + self.eps)
        return gx, gy, mag


    def _weighted_dice(self, probs, targets, weight):
        # 2*sum(w*p*t)/(sum(w*p)+sum(w*t))
        wp = (weight * probs).sum()
        wt = (weight * targets).sum()
        winter = (weight * probs * targets).sum()
        return 1.0 - (2.0 * winter + self.eps) / (wp + wt + self.eps)

    def _grad_loss_on_ring(self, probs, targets, ring):
        # 只在 ring 上计算幅值与方向一致性，避免被背景稀释
        gx_p, gy_p, mag_p = self._sobel(probs)
        gx_t, gy_t, mag_t = self._sobel(targets)

        # 避免 ring 为空
        denom = ring.sum().clamp_min(1.0)

        # 1) 幅值一致（L1）
        mag_l1 = ((mag_p - mag_t).abs() * ring).sum() / denom

        # 2) 方向一致（1 - cosine）
        cos = (gx_p*gx_t + gy_p*gy_t) / (mag_p*mag_t + self.eps)
        ori_loss = ((1.0 - cos) * ring).sum() / denom

        return (1.0 - self.weight_ori) * mag_l1 + self.weight_ori * ori_loss

    # ---------- 前向 ----------
    def forward(self, logits, targets):
        # 预处理
        targets = self._ensure_shape(targets, logits).float()
        probs   = torch.sigmoid(logits)

        # 边界权重图
        with torch.no_grad():
            ring = self._morph_grad(targets, self.ring_width)  # {0,1}
            weight = 1.0 + self.ring_alpha * ring

        # 子损失

        dice_loss = self._weighted_dice(probs, targets, weight)
        grad_loss = self._grad_loss_on_ring(probs, targets, ring)

        # 组合
        total = self.weight_dice * dice_loss \
                + self.weight_grad * grad_loss

        if self.return_components:
            return {
                "loss": total,
                "dice": dice_loss.detach(),

                "grad": grad_loss.detach()
            }
        return total


class BoundaryAwareLoss_grad(nn.Module):

    def __init__(
        self,

        ring_width: int = 2,
        ring_alpha: float = 6.0,
        weight_grad: float = 0.2,
        weight_ori: float  = 0.5,  # 方向一致性在梯度项中的占比（0~1）
        eps: float = 1e-6,
        return_components: bool = False,  # 需要时返回各子损失
    ):
        super().__init__()


        self.ring_width  = ring_width
        self.ring_alpha  = ring_alpha
        self.weight_grad = weight_grad
        self.weight_ori  = weight_ori
        self.eps = eps
        self.return_components = return_components



        # Sobel 核
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    # ---------- 辅助函数 ----------
    @staticmethod
    def _ensure_shape(x, like):
        # 保证 targets 形状是 [B,1,H,W]
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)
        if x.shape[2:] != like.shape[2:]:
            raise ValueError(f"size mismatch: {x.shape} vs {like.shape}")
        return x

    @staticmethod
    def _morph_grad(x, width):
        """ 近似形态学梯度：dilate - erode（用 max_pool 实现），再膨胀一圈保证连通 """
        dil = F.max_pool2d(x, kernel_size=2*width+1, stride=1, padding=width)
        ero = -F.max_pool2d(-x, kernel_size=2*width+1, stride=1, padding=width)
        edge = (dil - ero).clamp(0, 1)
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        return edge

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + self.eps)
        return gx, gy, mag



    def _grad_loss_on_ring(self, probs, targets, ring):
        # 只在 ring 上计算幅值与方向一致性，避免被背景稀释
        gx_p, gy_p, mag_p = self._sobel(probs)
        gx_t, gy_t, mag_t = self._sobel(targets)

        # 避免 ring 为空
        denom = ring.sum().clamp_min(1.0)

        # 1) 幅值一致（L1）
        mag_l1 = ((mag_p - mag_t).abs() * ring).sum() / denom

        # 2) 方向一致（1 - cosine）
        cos = (gx_p*gx_t + gy_p*gy_t) / (mag_p*mag_t + self.eps)
        ori_loss = ((1.0 - cos) * ring).sum() / denom

        return (1.0 - self.weight_ori) * mag_l1 + self.weight_ori * ori_loss

    # ---------- 前向 ----------
    def forward(self, logits, targets):
        # 预处理
        targets = self._ensure_shape(targets, logits).float()
        probs   = torch.sigmoid(logits)

        # 边界权重图
        with torch.no_grad():
            ring = self._morph_grad(targets, self.ring_width)  # {0,1}
            weight = 1.0 + self.ring_alpha * ring

        # 子损失


        grad_loss = self._grad_loss_on_ring(probs, targets, ring)

        # 组合
        total =  self.weight_grad * grad_loss

        if self.return_components:
            return {
                "loss": total,
                "grad": grad_loss.detach()
            }
        return total

class BoundaryAwareLoss_no_dice(nn.Module):
    """
    Dice + BCE（边界加权） + Sobel梯度一致性（只在GT边界环上计算）
    logits : [B,1,H,W]  未过 sigmoid
    targets: [B,1,H,W]  二值 {0,1}

    关键点：
    - ring_width 决定“边界环”宽度（像素）
    - ring_alpha 放大环上像素的权重，越大越“收边”
    - weight_grad 控制梯度一致性项权重
    - weight_ori 控制方向一致性（余弦）的子权重
    """
    def __init__(
        self,

        weight_bce: float = 0.5,
        ring_width: int = 2,
        ring_alpha: float = 6.0,
        weight_grad: float = 0.2,
        weight_ori: float  = 0.5,  # 方向一致性在梯度项中的占比（0~1）
        eps: float = 1e-6,
        return_components: bool = False,  # 需要时返回各子损失
    ):
        super().__init__()

        self.weight_bce  = weight_bce
        self.ring_width  = ring_width
        self.ring_alpha  = ring_alpha
        self.weight_grad = weight_grad
        self.weight_ori  = weight_ori
        self.eps = eps
        self.return_components = return_components

        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')

        # Sobel 核
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    # ---------- 辅助函数 ----------
    @staticmethod
    def _ensure_shape(x, like):
        # 保证 targets 形状是 [B,1,H,W]
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)
        if x.shape[2:] != like.shape[2:]:
            raise ValueError(f"size mismatch: {x.shape} vs {like.shape}")
        return x

    @staticmethod
    def _morph_grad(x, width):
        """ 近似形态学梯度：dilate - erode（用 max_pool 实现），再膨胀一圈保证连通 """
        dil = F.max_pool2d(x, kernel_size=2*width+1, stride=1, padding=width)
        ero = -F.max_pool2d(-x, kernel_size=2*width+1, stride=1, padding=width)
        edge = (dil - ero).clamp(0, 1)
        edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
        return edge

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + self.eps)
        return gx, gy, mag

    def _weighted_bce(self, logits, targets, weight):
        per_pix = self.bce_logits(logits, targets)  # [B,1,H,W]
        return (per_pix * weight).mean()


    def _grad_loss_on_ring(self, probs, targets, ring):
        # 只在 ring 上计算幅值与方向一致性，避免被背景稀释
        gx_p, gy_p, mag_p = self._sobel(probs)
        gx_t, gy_t, mag_t = self._sobel(targets)

        # 避免 ring 为空
        denom = ring.sum().clamp_min(1.0)

        # 1) 幅值一致（L1）
        mag_l1 = ((mag_p - mag_t).abs() * ring).sum() / denom

        # 2) 方向一致（1 - cosine）
        cos = (gx_p*gx_t + gy_p*gy_t) / (mag_p*mag_t + self.eps)
        ori_loss = ((1.0 - cos) * ring).sum() / denom

        return (1.0 - self.weight_ori) * mag_l1 + self.weight_ori * ori_loss

    # ---------- 前向 ----------
    def forward(self, logits, targets):
        # 预处理
        targets = self._ensure_shape(targets, logits).float()
        probs   = torch.sigmoid(logits)

        # 边界权重图
        with torch.no_grad():
            ring = self._morph_grad(targets, self.ring_width)  # {0,1}
            weight = 1.0 + self.ring_alpha * ring

        # 子损失
        bce_loss  = self._weighted_bce(logits, targets, weight)

        grad_loss = self._grad_loss_on_ring(probs, targets, ring)

        # 组合
        total = self.weight_bce * bce_loss \
                + self.weight_grad * grad_loss

        if self.return_components:
            return {
                "loss": total,
                "bce":  bce_loss.detach(),
                "grad": grad_loss.detach()
            }
        return total


import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from typing import Iterable, List, Set, Tuple

# ---------- Utils ----------
def simplex(t: Tensor, axis=1, atol: float = 1e-4) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, atol=atol, rtol=0)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

# ---------- Representation switches ----------
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # (B, C, H, W)
    assert simplex(probs)
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)
    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    """
    seg: (H,W) or (B,H,W) or (B,1,H,W) with integer labels in [0..C-1]
    returns: (B,C,H,W) int32 one-hot
    """
    if seg.dim() == 2:                 # (H, W)
        seg = seg.unsqueeze(0)         # -> (1, H, W)
    elif seg.dim() == 4 and seg.size(1) == 1:  # (B,1,H,W)
        seg = seg.squeeze(1)           # -> (B, H, W)
    assert seg.dim() == 3, f"Expected seg of shape (B,H,W), got {tuple(seg.shape)}"
    assert sset(seg, list(range(C))), "seg must be integer class labels in [0..C-1]"

    b, w, h = seg.shape  # (B, H, W)
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)  # (B,C,H,W)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)
    return res

# ---------- Distance transforms ----------
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    """
    seg: (C, H, W) numpy one-hot (0/1)
    return: (C, H, W) float32 signed distance map
    Outside object: positive; inside: negative (minus 1).
    """
    assert seg.ndim == 3, f"Expected (C,H,W), got {seg.shape}"
    assert one_hot(torch.tensor(seg), axis=0)
    C = seg.shape[0]

    res = np.zeros_like(seg, dtype=np.float32)
    for c in range(C):
        posmask = seg[c].astype(bool)  # fix deprecated np.bool
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def one_hot2dist_batch(seg: Tensor) -> Tensor:
    """
    seg: (B, C, H, W) torch one-hot (0/1)
    return: (B, C, H, W) float32 distance maps
    """
    assert one_hot(seg, axis=1), "seg must be valid one-hot along channel dim"
    B, C, H, W = seg.shape
    device = seg.device
    out = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
    for b in range(B):
        d_np = one_hot2dist(seg[b].detach().cpu().numpy())  # (C,H,W)
        out[b] = torch.from_numpy(d_np).to(device)
    return out

# ---------- Loss ----------
class SurfaceLoss():
    def __init__(self, idc: List[int] = None):
        # Only compute foreground by default for binary seg: idc=[1]
        self.idc: List[int] = [1] if idc is None else idc

    # probs: (B,C,H,W), dist_maps: (B,C,H,W)
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor = None) -> Tensor:
        assert simplex(probs), "probs must sum to 1 along channel dim"
        assert not one_hot(dist_maps), "dist_maps must be continuous distances, not one-hot"

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)
        loss = multiplied.mean()
        return loss

# ---------- Example for your shapes: (B=8, C=1, H=256, W=256) ----------
if __name__ == "__main__":
    B, H, W = 8, 256, 256

    # 模型输出（单通道 logit）：(B,1,H,W)
    logits = torch.randn(B, 1, H, W)

    # GT（二值标签，整型 0/1）：(B,1,H,W)
    y = (torch.rand(B, 1, H, W) > 0.7).long()

    # 1) 将单通道 logit -> 概率，并扩成两通道（满足 simplex）
    p = torch.sigmoid(logits)                # (B,1,H,W)
    probs = torch.cat([1.0 - p, p], dim=1)   # (B,2,H,W)  ch0=背景, ch1=前景

    # 2) 生成 one-hot 标签与距离图（批量版）
    with torch.no_grad():
        target_oh = class2one_hot(y, C=2)          # (B,2,H,W)
        dist_maps = one_hot2dist_batch(target_oh)  # (B,2,H,W)

    # 3) 计算 SurfaceLoss（默认只计前景通道 idc=[1]）
    criterion = SurfaceLoss()
    loss = criterion(probs, dist_maps, y)
    print("loss:", loss.item())
