import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Callable, Union, Tuple

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Global configuration
groups = 32


# ============================================================
# Utility Functions for Model
# ============================================================

def softmax(x: torch.Tensor, dim: Union[int, Tuple[int, ...]], temperature: float = 1.0) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions with temperature scaling.
    """
    x = x / temperature
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# ============================================================
# True Complementary Fusion Modules
# ============================================================

class FeatureDifferenceDetector(nn.Module):
    """特征差异检测器"""

    def __init__(self, dim_s, dim_l, scale):
        super().__init__()
        self.scale = scale

        # 通道对齐卷积
        align_channels = 64
        self.channel_align_e = nn.Conv2d(dim_s, align_channels, 1)
        self.channel_align_r = nn.Conv2d(dim_l, align_channels, 1)

        # 简化的差异检测网络
        self.difference_net = nn.Sequential(
            nn.Conv2d(align_channels * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, e, r):
        # 首先对齐通道维度
        e_aligned = self.channel_align_e(e)
        r_aligned = self.channel_align_r(r)

        # 调整到相同空间尺度
        target_size = e_aligned.shape[2:]

        # 调整r的尺寸以匹配e
        if r_aligned.shape[2:] != target_size:
            r_adjusted = F.interpolate(r_aligned, size=target_size, mode='bilinear', align_corners=True)
        else:
            r_adjusted = r_aligned

        # 计算特征差异
        combined_features = torch.cat([e_aligned, r_adjusted], dim=1)
        diff_map = self.difference_net(combined_features)

        return diff_map


class ComplementRegionDetector(nn.Module):
    """互补区域检测器 - 识别需要互补的空间区域"""

    def __init__(self, dim_s, dim_l, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # 多尺度特征差异检测
        self.difference_detectors = nn.ModuleList([
            FeatureDifferenceDetector(dim_s, dim_l, scale)
            for scale in [1, 2, 4]
        ])

        # 修复：确保中间通道数合理
        mid_channels = max(2, num_scales // 2)
        mid_channels = int(mid_channels)

        # 互补区域融合
        self.complement_fusion = nn.Sequential(
            nn.Conv2d(num_scales, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, e, r, level):
        differences = []

        # 在不同尺度上检测特征差异
        for detector in self.difference_detectors:
            diff_map = detector(e, r)
            differences.append(diff_map)

        # 确保所有差异图具有相同的空间尺寸
        target_size = e.shape[2:]

        aligned_differences = []
        for diff_map in differences:
            if diff_map.shape[2:] != target_size:
                aligned_diff = F.interpolate(diff_map, size=target_size,
                                             mode='bilinear', align_corners=True)
                aligned_differences.append(aligned_diff)
            else:
                aligned_differences.append(diff_map)

        # 拼接差异图
        diff_concat = torch.cat(aligned_differences, dim=1)

        # 融合多尺度差异
        complement_masks = self.complement_fusion(diff_concat)

        # 分割为高精度和低精度的互补掩码
        comp_mask_high = complement_masks[:, 0:1]
        comp_mask_low = complement_masks[:, 1:2]

        return comp_mask_high, comp_mask_low


class SpatialCoverageMask(nn.Module):
    """空间覆盖掩码 - 确保不同粒度的特征覆盖不同空间区域"""

    def __init__(self, dim_s, dim_l, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        # 高精度空间注意力
        self.high_precision_att = nn.Sequential(
            nn.Conv2d(dim_s, max(1, dim_s // 4), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim_s // 4), 1, 1),
            nn.Sigmoid()
        )

        # 低精度空间注意力
        self.low_precision_att = nn.Sequential(
            nn.Conv2d(dim_l, max(1, dim_l // 4), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim_l // 4), 1, 1),
            nn.Sigmoid()
        )

        # 互补区域检测
        self.complement_region = ComplementRegionDetector(dim_s, dim_l)

    def forward(self, e, r, level):
        # 高精度关注区域
        high_att = self.high_precision_att(e)

        # 低精度关注区域
        target_size = e.shape[2:]
        if r.shape[2:] != target_size:
            r_resized = F.interpolate(r, size=target_size, mode='bilinear', align_corners=True)
        else:
            r_resized = r

        low_att = self.low_precision_att(r_resized)

        # 检测互补区域
        comp_mask_high, comp_mask_low = self.complement_region(e, r, level)

        # 确保互补掩码与注意力图具有相同的空间尺寸
        if comp_mask_high.shape[2:] != target_size:
            comp_mask_high = F.interpolate(comp_mask_high, size=target_size,
                                           mode='bilinear', align_corners=True)
        if comp_mask_low.shape[2:] != target_size:
            comp_mask_low = F.interpolate(comp_mask_low, size=target_size,
                                          mode='bilinear', align_corners=True)

        # 确保空间覆盖互补
        coverage_mask_e = (1 - high_att) * comp_mask_high
        coverage_mask_r = (1 - low_att) * comp_mask_low

        return coverage_mask_e, coverage_mask_r


class ComplementaryAttention(nn.Module):
    """互补注意力机制 - 实现真正的特征互补"""

    def __init__(self, dim_s, dim_l, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        # 通道对齐
        align_channels = 64
        self.channel_align_e = nn.Conv2d(dim_s, align_channels, 1)
        self.channel_align_r = nn.Conv2d(dim_l, align_channels, 1)

        # 高精度到低精度的互补
        self.high_to_low = nn.Sequential(
            nn.Conv2d(align_channels, dim_l, 3, padding=1),
            nn.GroupNorm(min(groups, dim_l), dim_l),
            nn.ReLU(inplace=True)
        )

        # 低精度到高精度的互补
        self.low_to_high = nn.Sequential(
            nn.Conv2d(align_channels, dim_s, 3, padding=1),
            nn.GroupNorm(min(groups, dim_s), dim_s),
            nn.ReLU(inplace=True)
        )

        # 互补权重学习
        self.complement_weight = nn.Parameter(torch.ones(2))

        # 空间覆盖掩码
        self.spatial_mask = SpatialCoverageMask(dim_s, dim_l, scale_factor)

    def forward(self, e, r, level):
        # 首先对齐通道
        e_aligned = self.channel_align_e(e)
        r_aligned = self.channel_align_r(r)

        # 记录原始尺寸
        original_r_size = r.shape[2:]  # [H, W]

        # 确保输入尺寸匹配
        target_size = e_aligned.shape[2:]
        if r_aligned.shape[2:] != target_size:
            r_aligned = F.interpolate(r_aligned, size=target_size, mode='bilinear', align_corners=True)

        # 生成互补特征
        e_complement = self.high_to_low(e_aligned)

        # 将高精度互补特征调整到低精度特征的原始尺寸
        if e_complement.shape[2:] != original_r_size:
            e_complement = F.interpolate(e_complement, size=original_r_size,
                                         mode='bilinear', align_corners=True)

        # 低精度补充高精度
        r_complement = self.low_to_high(r_aligned)

        # 空间覆盖掩码
        coverage_mask_e, coverage_mask_r = self.spatial_mask(e, r, level)

        # 确保所有张量空间尺寸匹配
        target_size_e = e.shape[2:]
        target_size_r = r.shape[2:]

        # 调整互补特征尺寸
        if r_complement.shape[2:] != target_size_e:
            r_complement = F.interpolate(r_complement, size=target_size_e,
                                         mode='bilinear', align_corners=True)
        if coverage_mask_e.shape[2:] != target_size_e:
            coverage_mask_e = F.interpolate(coverage_mask_e, size=target_size_e,
                                            mode='bilinear', align_corners=True)
        if coverage_mask_r.shape[2:] != target_size_r:
            coverage_mask_r = F.interpolate(coverage_mask_r, size=target_size_r,
                                            mode='bilinear', align_corners=True)

        # 应用互补
        e_enhanced = e + coverage_mask_e * r_complement * self.complement_weight[0]
        r_enhanced = r + coverage_mask_r * e_complement * self.complement_weight[1]

        return e_enhanced, r_enhanced


class CoverageEvaluator(nn.Module):
    """覆盖度评估器 - 评估特征的空间覆盖质量"""

    def __init__(self, dim):
        super().__init__()
        self.coverage_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(1, dim // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim // 4), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.coverage_net(x)


class GranularityAwareGate(nn.Module):
    """粒度感知门控 - 根据输入特性自适应调整互补强度"""

    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(1, dim // 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim // 2), 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate_values = self.gate_net(x)
        return gate_values[:, 0:1], gate_values[:, 1:2]


class SpatialCoverageFusion(nn.Module):
    """空间覆盖融合 - 整合不同粒度的互补特征"""

    def __init__(self, dim_s, dim_l):
        super().__init__()
        # 确保中间通道数合理
        mid_channels = (dim_s + dim_l) // 2
        mid_channels = max(groups, mid_channels)  # 至少为groups
        if mid_channels % groups != 0:
            mid_channels = (mid_channels // groups) * groups

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim_s + dim_l, mid_channels, 3, padding=1),
            nn.GroupNorm(min(groups, mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim_l, 3, padding=1),
            nn.GroupNorm(min(groups, dim_l), dim_l),
            nn.ReLU(inplace=True)
        )

        # 覆盖度评估
        self.coverage_evaluator = CoverageEvaluator(dim_l)

    def forward(self, complemented_features):
        # 整合所有互补特征
        all_features = []

        for e_comp, r_comp in complemented_features:
            # 上采样低精度特征以匹配高精度
            r_upsampled = F.interpolate(r_comp, size=e_comp.shape[2:],
                                        mode='bilinear', align_corners=True)
            fused = torch.cat([e_comp, r_upsampled], dim=1)
            all_features.append(fused)

        # 多尺度融合
        if len(all_features) > 1:
            pyramid_fused = self.pyramid_fusion(all_features)
        else:
            pyramid_fused = all_features[0]

        # 最终融合
        final_fused = self.fusion_conv(pyramid_fused)

        # 评估覆盖度
        coverage_score = self.coverage_evaluator(final_fused)

        return final_fused

    def pyramid_fusion(self, features):
        """金字塔式特征融合"""
        if not features:
            return None

        fused = features[0]
        for i in range(1, len(features)):
            # 调整尺寸匹配
            if features[i].shape[2:] != fused.shape[2:]:
                features[i] = F.interpolate(features[i], size=fused.shape[2:],
                                            mode='bilinear', align_corners=True)
            # 加权融合
            weight = 1.0 / (i + 1)
            fused = weight * features[i] + (1 - weight) * fused

        return fused


class HierarchicalComplementaryFusion(nn.Module):
    """
    层次化互补融合模块 - 实现真正的分支互补
    """

    def __init__(self, dim_s, dim_l, levels=4):
        super().__init__()
        self.levels = levels
        self.dim_l = dim_l

        # 多尺度互补注意力机制
        self.complementary_attentions = nn.ModuleList()
        for i in range(levels):
            scale_factor = 2 ** i
            self.complementary_attentions.append(
                ComplementaryAttention(dim_s, dim_l, scale_factor)
            )

        # 空间覆盖融合
        self.spatial_coverage_fusion = SpatialCoverageFusion(dim_s, dim_l)

        # 粒度感知门控
        self.granularity_gate = GranularityAwareGate(dim_s + dim_l)

    def forward(self, e, r):
        """
        e: 高精度特征 (精细粒度)
        r: 低精度特征 (粗糙粒度)
        """
        complemented_features = []

        # 在不同尺度上进行互补
        for i, comp_att in enumerate(self.complementary_attentions):
            e_comp, r_comp = comp_att(e, r, level=i)
            complemented_features.append((e_comp, r_comp))

        # 空间覆盖融合
        fused_feature = self.spatial_coverage_fusion(complemented_features)

        return fused_feature


# ============================================================
# Basic Modules
# ============================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads
        k = self.to_k(x_qkv)
        k = k.reshape(b, n, h, -1).transpose(1, 2)
        v = self.to_v(x_qkv)
        v = v.reshape(b, n, h, -1).transpose(1, 2)
        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = q.reshape(b, 1, h, -1).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, 1, -1)
        out = self.to_out(out)
        return out


# ============================================================
# Mixture of Experts (MoE) Layer
# ============================================================

class SoftMoELayerWrapper(nn.Module):
    """
    An enhanced wrapper class to create a Soft Mixture of Experts layer.
    """

    def __init__(
            self,
            dim: int,
            num_experts: int,
            slots_per_expert: int,
            layer: Callable,
            normalize: bool = True,
            router_temp: float = 0.1,
            aux_loss_weight: float = 0.08,
            z_loss_weight: float = 0.001,
            add_residual: bool = True,
            **layer_kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize
        self.router_temp = router_temp
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.add_residual = add_residual

        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))
        nn.init.xavier_normal_(self.phi, gain=1.5)

        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens_processed', torch.tensor(0.0))

    def _compute_balance_loss(self, dispatch_weights, combine_weights):
        expert_usage = torch.mean(torch.sum(dispatch_weights, dim=1), dim=0)
        expert_usage = torch.sum(expert_usage, dim=1)

        if self.training:
            with torch.no_grad():
                batch_tokens = float(dispatch_weights.size(0) * dispatch_weights.size(1))
                self.expert_usage.add_(
                    torch.sum(dispatch_weights.reshape(-1, self.num_experts, self.slots_per_expert), dim=(0, 2)))
                self.total_tokens_processed.add_(batch_tokens)

        expert_usage = expert_usage / expert_usage.sum()
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        balance_loss = F.kl_div(expert_usage.log(), target_usage, reduction='sum')
        return balance_loss

    def _compute_router_z_loss(self, logits):
        log_z = torch.logsumexp(logits, dim=(2, 3))
        z_loss = torch.mean(log_z ** 2)
        return z_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim, f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert len(x.shape) == 3, f"Input expected to have 3 dimensions but has {len(x.shape)}"

        identity = x
        phi = self.phi

        if self.normalize:
            x_normalized = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(phi, dim=0)
        else:
            x_normalized = x

        # 使用einsum进行矩阵乘法
        logits = torch.einsum("bmd,dnp->bmnp", x_normalized, phi)
        d = softmax(logits, dim=1, temperature=self.router_temp)
        c = softmax(logits, dim=(2, 3), temperature=self.router_temp)

        aux_loss = 0.0
        if self.training:
            aux_loss = self.aux_loss_weight * self._compute_balance_loss(d, c)
            aux_loss = aux_loss + self.z_loss_weight * self._compute_router_z_loss(logits)
            self._moe_loss = aux_loss

        xs = torch.einsum("bmd,bmnp->bnpd", x, d)
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        if self.add_residual:
            y = y + identity

        return y

    def get_moe_loss(self):
        return getattr(self, '_moe_loss', torch.tensor(0.0))

    def get_expert_usage_stats(self):
        if self.total_tokens_processed > 0:
            normalized_usage = self.expert_usage / self.total_tokens_processed
            return normalized_usage
        return torch.zeros_like(self.expert_usage)


# ============================================================
# Swin Transformer Components
# ============================================================

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # 确保mask与attn在同一个设备上
            mask = mask.to(attn.device)  # 添加这行
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block."""

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_moe=True,
            num_experts=8,
            slots_per_expert=2,
            router_temp=0.05,
            aux_loss_weight=0.08,
            z_loss_weight=0.001):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_moe = use_moe
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_moe:
            def mlp_factory(**kwargs):
                return Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop=drop,
                    **kwargs
                )

            self.mlp = SoftMoELayerWrapper(
                dim=dim,
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
                layer=mlp_factory,
                normalize=True,
                router_temp=router_temp,
                aux_loss_weight=aux_loss_weight,
                z_loss_weight=z_loss_weight,
                add_residual=False
            )
            self.register_buffer('moe_loss', torch.tensor(0.0))
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W

        if L != H * W:
            H = int(np.sqrt(L))
            W = L // H
            self.H, self.W = H, W

        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H * W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 确保mask_matrix在正确的设备上
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)  # 添加这行

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        if self.use_moe and self.training:
            identity = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = identity + self.drop_path(x)
            if hasattr(self, 'moe_loss'):
                self.moe_loss = self.mlp.get_moe_loss()
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def get_moe_loss(self):
        if self.use_moe and hasattr(self, 'moe_loss'):
            return self.moe_loss
        return torch.tensor(0.0)


class PatchMerging(nn.Module):
    """ Patch Merging Layer"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        Wh, Ww = x.size(2), x.size(3)

        if self.norm is not None:
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.norm(x_flat)
            x = x_flat.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x, (Wh, Ww)


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage."""

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 up=True,
                 use_moe=True,
                 num_experts=4,
                 slots_per_expert=1,
                 moe_layer_indices=None,
                 router_temp=0.1,
                 aux_loss_weight=0.08,
                 z_loss_weight=0.001):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up
        self.use_moe = use_moe
        self.dim = dim

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            use_moe_in_block = False
            if use_moe and (moe_layer_indices is None or i in moe_layer_indices):
                use_moe_in_block = True

            block = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_moe=use_moe_in_block,
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
                router_temp=router_temp,
                aux_loss_weight=aux_loss_weight,
                z_loss_weight=z_loss_weight)
            self.blocks.append(block)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        if use_moe:
            self.register_buffer('moe_loss', torch.tensor(0.0))

    def forward(self, x, H, W):
        B, L, C = x.shape
        if L != H * W:
            new_H = int(np.sqrt(L))
            new_W = L // new_H
            H, W = new_H, new_W

        moe_loss = 0.0
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1))
        shift_size_int = int(self.shift_size)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size_int),
                    slice(-shift_size_int, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size_int),
                    slice(-shift_size_int, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
            if self.use_moe and self.training:
                moe_loss = moe_loss + blk.get_moe_loss()

        if self.use_moe and self.training and hasattr(self, 'moe_loss'):
            self.moe_loss = moe_loss

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if self.up:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            else:
                Wh, Ww = H * 2, W * 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def get_moe_loss(self):
        if self.use_moe and hasattr(self, 'moe_loss'):
            return self.moe_loss
        return torch.tensor(0.0)


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone."""

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 use_moe=True,
                 num_experts=4,
                 slots_per_expert=1,
                 moe_stages=None,
                 moe_layers_each_stage=None,
                 router_temp=0.1,
                 aux_loss_weight=0.08,
                 z_loss_weight=0.001,
                 moe_loss_weight=0.1):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.use_moe = use_moe
        self.moe_stages = moe_stages
        self.moe_loss_weight = moe_loss_weight

        if use_moe:
            self.register_buffer('moe_aux_loss', torch.tensor(0.0))

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            use_moe_in_stage = False
            moe_layer_indices = None
            if use_moe:
                if moe_stages is None or i_layer in moe_stages:
                    use_moe_in_stage = True
                    if moe_layers_each_stage is not None and i_layer < len(moe_layers_each_stage):
                        moe_layer_indices = moe_layers_each_stage[i_layer]

            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_moe=use_moe_in_stage,
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
                moe_layer_indices=moe_layer_indices,
                router_temp=router_temp,
                aux_loss_weight=aux_loss_weight,
                z_loss_weight=z_loss_weight)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i in out_indices:
            layer = norm_layer(num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            try:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
                if missing_keys:
                    print(f'缺失的键: {missing_keys}')
                if unexpected_keys:
                    print(f'意外的键: {unexpected_keys}')
                print(f'从 {pretrained} 加载预训练权重成功')
            except Exception as e:
                print(f'从 {pretrained} 加载预训练权重失败: {e}')
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained 必须是字符串或 None')

    def forward(self, x):
        if self.use_moe and self.training and hasattr(self, 'moe_aux_loss'):
            self.moe_aux_loss = torch.tensor(0.0)

        x, hw_shape = self.patch_embed(x)
        Wh, Ww = hw_shape

        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)
        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if self.use_moe and self.training and hasattr(self, 'moe_aux_loss'):
                if self.moe_stages is None or i in self.moe_stages:
                    if hasattr(layer, 'get_moe_loss'):
                        self.moe_aux_loss = self.moe_aux_loss + layer.get_moe_loss()

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def get_moe_loss(self):
        if self.use_moe and hasattr(self, 'moe_aux_loss'):
            return self.moe_aux_loss * self.moe_loss_weight
        return torch.tensor(0.0)

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


# ============================================================
# Decoder Components
# ============================================================

class up_conv(nn.Module):
    """Up Convolution Block"""

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class conv_block(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Conv_block(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = up_conv(in_channels, out_channels)
        # 修正：输入通道数应该是 skip_channels + out_channels
        self.conv_relu = nn.Sequential(
            nn.Conv2d(skip_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 确保空间尺寸匹配
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        # 拼接特征图
        x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ============================================================
# Main UNet Model with True Complementary Fusion
# ============================================================
class UNetWithTrueComplement(nn.Module):
    def __init__(self,
                 dim,
                 n_class,
                 in_ch=3,
                 use_moe=False,
                 num_experts=4,
                 slots_per_expert=1,
                 moe_stages=None,
                 moe_layers_each_stage=None,
                 router_temp=0.1,
                 aux_loss_weight=0.08,
                 z_loss_weight=0.001,
                 moe_loss_weight=0.1):
        super().__init__()
        self.use_moe = use_moe
        self.moe_loss_weight = moe_loss_weight

        if use_moe:
            self.num_experts = num_experts
            self.slots_per_expert = slots_per_expert
            self.moe_stages = moe_stages if moe_stages is not None else "[all]"
            self.moe_layers_each_stage = moe_layers_each_stage
            self.router_temp = router_temp
            self.aux_loss_weight = aux_loss_weight
            self.z_loss_weight = z_loss_weight
            self.register_buffer('moe_aux_loss', torch.tensor(0.0))

        # 第一个编码器：较大的模型，通道数为128
        self.encoder = SwinTransformer(
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            drop_path_rate=0.5,
            embed_dim=128,
            use_moe=use_moe,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            moe_stages=moe_stages,
            moe_layers_each_stage=moe_layers_each_stage,
            router_temp=router_temp,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
            moe_loss_weight=moe_loss_weight
        )

        # 第二个编码器：较小的模型，通道数为96
        self.encoder2 = SwinTransformer(
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.2,
            patch_size=8,
            embed_dim=96,
            use_moe=use_moe,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            moe_stages=moe_stages,
            moe_layers_each_stage=moe_layers_each_stage,
            router_temp=router_temp,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
            moe_loss_weight=moe_loss_weight
        )

        # 初始化权重
        self.encoder.init_weights(None)
        self.encoder2.init_weights(None)

        # 修正互补融合模块的通道数
        self.complement_fusion_1 = HierarchicalComplementaryFusion(96, 128)
        self.complement_fusion_2 = HierarchicalComplementaryFusion(192, 256)
        self.complement_fusion_3 = HierarchicalComplementaryFusion(384, 512)
        self.complement_fusion_4 = HierarchicalComplementaryFusion(768, 1024)

        # 修复：调整融合卷积的通道数
        self.fusion_conv1 = Conv_block(128, 128)
        self.fusion_conv2 = Conv_block(256, 256)
        self.fusion_conv3 = Conv_block(512, 512)
        self.fusion_conv4 = Conv_block(1024, 1024)

        # Decoder layers - 修正通道数匹配问题
        self.layer1 = Decoder(1024, 512, 512)
        self.layer2 = Decoder(512, 256, 256)
        self.layer3 = Decoder(256, 128, 128)
        self.layer4 = Decoder(128, 64, 64)
        self.layer5 = Decoder(64, 32, 32)

        # 下采样层
        self.down1 = nn.Conv2d(in_ch, 32, kernel_size=1, stride=1, padding=0)
        self.down2 = conv_block(32, 64)

        # 最终输出层
        self.final = nn.Conv2d(32, n_class, kernel_size=1, stride=1, padding=0)

        # 修复辅助损失层 - 动态计算上采样倍数
        self.loss1 = nn.Sequential(
            nn.Conv2d(1024, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.loss2 = nn.Sequential(
            nn.Conv2d(128, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        if self.use_moe and self.training and hasattr(self, 'moe_aux_loss'):
            self.moe_aux_loss = torch.tensor(0.0)

        # 获取输入尺寸
        input_size = x.shape[2:]

        # 编码器前向传播
        out = self.encoder(x)
        out2 = self.encoder2(x)

        if self.use_moe and self.training and hasattr(self, 'moe_aux_loss'):
            if hasattr(self.encoder, 'get_moe_loss'):
                self.moe_aux_loss = self.moe_aux_loss + self.encoder.get_moe_loss()
            if hasattr(self.encoder2, 'get_moe_loss'):
                self.moe_aux_loss = self.moe_aux_loss + self.encoder2.get_moe_loss()

        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]
        r1, r2, r3, r4 = out2[0], out2[1], out2[2], out2[3]

        # 使用真正的互补融合
        e1 = self.complement_fusion_1(r1, e1)
        e2 = self.complement_fusion_2(r2, e2)
        e3 = self.complement_fusion_3(r3, e3)
        e4 = self.complement_fusion_4(r4, e4)

        # 应用融合卷积
        e1 = self.fusion_conv1(e1)
        e2 = self.fusion_conv2(e2)
        e3 = self.fusion_conv3(e3)
        e4 = self.fusion_conv4(e4)

        # 辅助损失 - 动态上采样到输入尺寸
        loss1_out = self.loss1(e4)
        loss1 = F.interpolate(loss1_out, size=input_size, mode='bilinear', align_corners=True)

        # 下采样路径
        ds1 = self.down1(x)
        ds2 = self.down2(ds1)

        # 解码器路径
        d1 = self.layer1(e4, e3)
        d2 = self.layer2(d1, e2)
        d3 = self.layer3(d2, e1)

        loss2_out = self.loss2(d3)
        loss2 = F.interpolate(loss2_out, size=input_size, mode='bilinear', align_corners=True)

        d4 = self.layer4(d3, ds2)
        d5 = self.layer5(d4, ds1)

        # 最终输出
        o = self.final(d5)

        if self.use_moe and self.training:
            return o, loss1, loss2, self.moe_aux_loss
        else:
            return o, loss1, loss2


# ============================================================
# 使用示例和训练代码
# ============================================================

def create_model(config):
    """创建模型实例"""
    model = UNetWithTrueComplement(
        dim=128,
        n_class=config['num_classes'],
        in_ch=config['in_channels'],
        use_moe=config.get('use_moe', False),
        num_experts=config.get('num_experts', 4),
        slots_per_expert=config.get('slots_per_expert', 1),
        moe_stages=config.get('moe_stages', None),
        moe_layers_each_stage=config.get('moe_layers_each_stage', None),
        router_temp=config.get('router_temp', 0.1),
        aux_loss_weight=config.get('aux_loss_weight', 0.08),
        z_loss_weight=config.get('z_loss_weight', 0.001),
        moe_loss_weight=config.get('moe_loss_weight', 0.1)
    )
    return model


def calculate_total_loss(outputs, targets, config):
    """计算总损失"""
    if len(outputs) == 4:
        pred, aux1, aux2, moe_loss = outputs
    else:
        pred, aux1, aux2 = outputs
        moe_loss = torch.tensor(0.0)

    # 主损失
    main_loss = structure_loss(pred, targets)

    # 辅助损失
    aux1_loss = structure_loss(aux1, targets)
    aux2_loss = structure_loss(aux2, targets)

    # 总损失
    total_loss = (main_loss +
                  config['aux_weight1'] * aux1_loss +
                  config['aux_weight2'] * aux2_loss +
                  moe_loss)

    return total_loss, main_loss, aux1_loss, aux2_loss, moe_loss


class UNetTrainer:
    """UNet训练器"""

    def __init__(self, config):
        self.config = config
        self.model = create_model(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(images)

            # 计算损失
            loss, main_loss, aux1_loss, aux2_loss, moe_loss = calculate_total_loss(
                outputs, masks, self.config
            )

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config['log_interval'] == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Main: {main_loss.item():.4f}, Aux1: {aux1_loss.item():.4f}, '
                      f'Aux2: {aux2_loss.item():.4f}, MoE: {moe_loss.item():.4f}')

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in dataloader:
                outputs = self.model(images)
                loss, _, _, _, _ = calculate_total_loss(outputs, masks, self.config)
                total_loss += loss.item()

        return total_loss / len(dataloader)


if __name__ == "__main__":
    # 配置参数
    config = {
        'num_classes': 1,
        'in_channels': 3,
        'use_moe': True,
        'num_experts': 4,
        'slots_per_expert': 1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'aux_weight1': 0.4,
        'aux_weight2': 0.4,
        'log_interval': 10
    }

    # 创建模型
    model = create_model(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)

    if len(outputs) == 4:
        pred, aux1, aux2, moe_loss = outputs
        print(f"输出形状 - 主输出: {pred.shape}, 辅助1: {aux1.shape}, 辅助2: {aux2.shape}")
        print(f"MoE损失: {moe_loss}")
    else:
        pred, aux1, aux2 = outputs
        print(f"输出形状 - 主输出: {pred.shape}, 辅助1: {aux1.shape}, 辅助2: {aux2.shape}")