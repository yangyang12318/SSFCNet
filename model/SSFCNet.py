import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: padding helpers
# -----------------------------
def _pad_to_multiple(x, multiple=16, mode='reflect'):
    B, C, H, W = x.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    if pad_h or pad_w:
        x = F.pad(x, (pl, pr, pt, pb), mode=mode)
    return x, (pl, pr, pt, pb)

def _unpad(x, pad):
    pl, pr, pt, pb = pad
    if (pl or pr or pt or pb):
        return x[:, :, pt:x.shape[2]-pb, pl:x.shape[3]-pr]
    return x


# -----------------------------
# Haar DWT / IWT (fixed kernels)
# -----------------------------
def dwt_init(x):
    # x: [B,C,H,W]
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_LH, x_HL, x_HH

def iwt_init(LL, LH, HL, HH):
    # make iwt(dwt(x)) ≈ x  (consistent scaling)
    r = 2
    B, C, H, W = LL.size()
    out = torch.zeros([B, C, r*H, r*W], device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = LL - LH - HL + HH
    out[:, :, 1::2, 0::2] = LL - LH + HL - HH
    out[:, :, 0::2, 1::2] = LL + LH - HL - HH
    out[:, :, 1::2, 1::2] = LL + LH + HL + HH
    return 0.5 * out


# -----------------------------
# Norm helper (GN default)
# -----------------------------
def _choose_gn_groups(c):
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

def _norm2d(c, kind='gn'):
    if kind == 'bn':
        return nn.BatchNorm2d(c)
    return nn.GroupNorm(num_groups=_choose_gn_groups(c), num_channels=c)



class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))



# -----------------------------
# Wavelet Encoder Block (V2)
# -----------------------------
class WaveletEncoderBlockV2(nn.Module):
    """
    输入:  [B, C, H, W]
    输出:  [B, Cout, H/2, W/2]
    - 定向增强: LH(1×k), HL(k×1), HH(dilated k×k)
    - 频带内 SE + 跨频带空间门
    - 融合 + 残差下采样（AvgPool2d + 1×1）
    """
    def __init__(self, in_channels, out_channels, k=3, reduction=16,
                 norm='gn', padding_mode='reflect'):
        super().__init__()
        Norm = lambda c: _norm2d(c, norm)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            Norm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            Norm(in_channels), nn.ReLU(inplace=True),
        )
        correct_padding = (k - 1) * 2 // 2
        # Orientation-aware depthwise conv on high-frequency bands
        self.dw_lh = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k),
                               padding=(0, k//2), groups=in_channels, bias=False)
        self.dw_hl = nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1),
                               padding=(k//2, 0), groups=in_channels, bias=False)
        self.dw_hh = nn.Conv2d(in_channels, in_channels, kernel_size=k,
                               padding=correct_padding, dilation=2, groups=in_channels, bias=False)

        # Band-wise SE
        cr = max(1, in_channels // reduction)
        def _se(c):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, cr, 1, bias=True), nn.ReLU(inplace=True),
                nn.Conv2d(cr, c, 1, bias=True), nn.Sigmoid()
            )
        self.se_ll = _se(in_channels)
        self.se_lh = _se(in_channels)
        self.se_hl = _se(in_channels)
        self.se_hh = _se(in_channels)

        # HF compress: 3C -> C
        self.hf_compress = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            Norm(in_channels)
        )

        # Spatial gate over all bands
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            Norm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1, bias=True), nn.Sigmoid()
        )

        # Fuse LL + HF -> out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size=3,
                      padding=1, padding_mode=padding_mode, bias=False),
            Norm(out_channels), nn.ReLU(inplace=True)
        )

        # Residual downsample skip
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            Norm(out_channels)
        )

    def forward(self, x):
        x = self.pre(x)
        LL, LH, HL, HH = dwt_init(x)  # [B,C,H/2,W/2]

        # Orientation-aware enhancement
        LH = self.dw_lh(LH)
        HL = self.dw_hl(HL)
        HH = self.dw_hh(HH)

        # Band-wise SE
        #LL = LL * self.se_ll(LL)
        LH = LH * self.se_lh(LH)
        HL = HL * self.se_hl(HL)
        HH = HH * self.se_hh(HH)

        # Spatial gate
        gate = self.spatial_gate(torch.cat([ LH, HL, HH], dim=1))
        gHF =  gate

        HF = self.hf_compress(torch.cat([LH, HL, HH], dim=1)) * gHF

        out = self.fuse(HF)
        out = out + self.skip(x)  # residual downsample
        return out


# -----------------------------
# Up-sampling blocks
# -----------------------------
class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='gn', padding_mode='reflect'):
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            _norm2d(out_channels, norm), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            _norm2d(out_channels, norm), nn.ReLU(inplace=True),
        )

class IWTUp(nn.Module):
    """Learn [LL,LH,HL,HH] via 1x1 conv, apply IWT, then refine."""
    def __init__(self, in_channels, out_channels, norm='gn', padding_mode='reflect'):
        super().__init__()
        self.to_bands = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=True)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            _norm2d(out_channels, norm), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      padding_mode=padding_mode, bias=False),
            _norm2d(out_channels, norm), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        bands = self.to_bands(x)
        LL, LH, HL, HH = torch.chunk(bands, 4, dim=1)
        up = iwt_init(LL, LH, HL, HH)  # [B,out,2H,2W]
        return self.refine(up)


# -----------------------------
# Diff-corr helper (|a-b| and a*b)
# -----------------------------
class DiffCorrProject(nn.Module):
    """Project cat(|a-b|, a*b) from 2*C -> C' (usually C'=C)."""
    def __init__(self, in_ch, out_ch, norm='gn'):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            _norm2d(out_ch, norm), nn.ReLU(inplace=True)
        )
    def forward(self, a, b):
        x = torch.cat([torch.abs(a - b), a * b], dim=1)
        return self.proj(x)

class cross_fuse_3d(nn.Module):
    def __init__(self, in_channels,norm='gn'):
        super(cross_fuse_3d, self).__init__()
        self.conv3d1 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,3,3], stride=1, padding=1, bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))
        self.conv3d2 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,5,5], stride=1, padding=(1, 2, 2), bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))
        self.conv3d3 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=[3,7,7], stride=1, padding=(1, 3, 3), bias=True),
                                nn.ReLU(),
                                nn.BatchNorm3d(1))

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            _norm2d(in_channels, norm),
        )

        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):
        cross_x = x.unsqueeze(1)
        cross_x1 = self.conv3d1(cross_x)
        cross_x1 = cross_x1.squeeze(1)
        cross_x2 = self.conv3d2(cross_x)
        cross_x2 = cross_x2.squeeze(1)
        cross_x3 = self.conv3d3(cross_x)
        cross_x3 = cross_x3.squeeze(1)
        cross_x = torch.cat((cross_x1, cross_x2, cross_x3), dim=1)
        cross_x = self.fuse_conv(cross_x)
        cross_x = self.dropout(cross_x)

        return cross_x

# -----------------------------
# O (full)
# -----------------------------
class SSFCNet(nn.Module):

    def __init__(self,
                 n_classes=1,
                 shared_encoder=True,
                 use_iwt_decoder=True,
                 use_diffcorr=True,

                 norm='gn',
                 padding_mode='reflect',
                 auto_pad=True):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.use_iwt_decoder = use_iwt_decoder
        self.use_diffcorr = use_diffcorr
        self.auto_pad = auto_pad

        Up = IWTUp if use_iwt_decoder else UpConvBlock

        # -------- Encoders --------
        self.enc0_a = WaveletEncoderBlockV2(3,   64, norm=norm, padding_mode=padding_mode)
        self.enc1_a = WaveletEncoderBlockV2(64,  128, norm=norm, padding_mode=padding_mode)
        self.enc2_a = WaveletEncoderBlockV2(128, 256, norm=norm, padding_mode=padding_mode)
        self.enc3_a = WaveletEncoderBlockV2(256, 512, norm=norm, padding_mode=padding_mode)

        if shared_encoder:

            self.enc0_b = self.enc0_a
            self.enc1_b = self.enc1_a
            self.enc2_b = self.enc2_a
            self.enc3_b = self.enc3_a
        else:
            self.enc0_b = WaveletEncoderBlockV2(3,   64, norm=norm, padding_mode=padding_mode)
            self.enc1_b = WaveletEncoderBlockV2(64,  128, norm=norm, padding_mode=padding_mode)
            self.enc2_b = WaveletEncoderBlockV2(128, 256, norm=norm, padding_mode=padding_mode)
            self.enc3_b = WaveletEncoderBlockV2(256, 512, norm=norm, padding_mode=padding_mode)



        self.cross0=cross_fuse_3d(64)
        self.cross1 = cross_fuse_3d(128)
        self.cross2 = cross_fuse_3d(256)
        self.cross3 = cross_fuse_3d(512)

        # -------- Diff-corr projections --------
        if use_diffcorr:
            self.proj2 = DiffCorrProject(in_ch=2*256, out_ch=256, norm=norm)
            self.proj1 = DiffCorrProject(in_ch=2*128, out_ch=128, norm=norm)
            self.proj0 = DiffCorrProject(in_ch=2*64,  out_ch=64,  norm=norm)

        # -------- Decoders --------
        # Stage 3 (deepest): cat(a3,b3)=1024 -> 512
        self.up4 = Up(1024, 512, norm=norm, padding_mode=padding_mode)

        # Stage 2: x(512)+a2(256)+b2(256)+diff2(256 if used)
        in_ch_up3 = 512 + 256 + 256 + (256 if use_diffcorr else 0)  # 1280 or 1024
        self.up3 = Up(in_ch_up3, 512, norm=norm, padding_mode=padding_mode)

        # Stage 1: x(512)+a1(128)+b1(128)+diff1(128)
        in_ch_up2 = 512 + 128 + 128 + (128 if use_diffcorr else 0)  # 896 or 768
        self.up2 = Up(in_ch_up2, 256, norm=norm, padding_mode=padding_mode)

        # Stage 0: x(256)+a0(64)+b0(64)+diff0(64)
        in_ch_up1 = 256 + 64 + 64 + (64 if use_diffcorr else 0)     # 448 or 384
        self.up1 = Up(in_ch_up1, 128, norm=norm, padding_mode=padding_mode)
        # -------- Head (logits) --------
        self.head = nn.Sequential(
            nn.Conv2d(128, n_classes, 3, padding=1, padding_mode=padding_mode, bias=True),
            nn.Conv2d(n_classes, n_classes, 1, padding=0, bias=True),
        )
    def forward(self, img1, img2):
        # Auto pad to multiples of 16
        if self.auto_pad:
            img1, pad1 = _pad_to_multiple(img1, 16, mode='reflect')
            img2, pad2 = _pad_to_multiple(img2, 16, mode='reflect')
        else:
            pad1 = pad2 = (0, 0, 0, 0)

        # ----- Encoder A -----
        a0 = self.enc0_a(img1)   # 64,  H/2
        a0 = a0 + self.cross0(a0)
        a1 = self.enc1_a(a0)     # 128, H/4
        a1 = a1 + self.cross1(a1)
        a2 = self.enc2_a(a1)     # 256, H/8
        a2 = a2 + self.cross2(a2)
        a3 = self.enc3_a(a2)     # 512, H/16
        a3 = a3 + self.cross3(a3)
        # ----- Encoder B -----

        b0 = self.enc0_b(img2)
        b0 =b0+self.cross0(b0)
        b1 = self.enc1_b(b0)
        b1 = b1 + self.cross1(b1)
        b2 = self.enc2_b(b1)
        b2 = b2 + self.cross2(b2)
        b3 = self.enc3_b(b2)
        b3 = b3 + self.cross3(b3)


        # ----- Decode & fuse -----
        x = torch.cat([a3, b3], dim=1)  # [B,1024,H/16,W/16]
        x = self.up4(x)                 # -> [B,512,H/8,W/8]

        # Stage 2
        feats = [x, a2, b2]
        if self.use_diffcorr:
            feats += [self.proj2(a2, b2)]
        x = torch.cat(feats, dim=1)
        x = self.up3(x)                 # -> [B,512,H/4,W/4]

        # Stage 1
        feats = [x, a1, b1]
        if self.use_diffcorr:
            feats += [self.proj1(a1, b1)]
        x = torch.cat(feats, dim=1)
        x = self.up2(x)                 # -> [B,256,H/2,W/2]

        # Stage 0
        feats = [x, a0, b0]
        if self.use_diffcorr:
            feats += [self.proj0(a0, b0)]
        x = torch.cat(feats, dim=1)
        x = self.up1(x)                 # -> [B,128,H,W]


        logits = self.head(x)         # logits (no sigmoid)

        # Unpad to original size
        if self.auto_pad:
            logits = _unpad(logits, pad1)  # pad1 == pad2 if both padded to same multiple

        return logits


