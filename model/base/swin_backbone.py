import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base.components import Conv


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """MLP模块，参考官方实现"""
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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding，参考官方实现"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # 移除严格的尺寸检查，允许灵活的输入尺寸
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA)，参考官方实现"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        
        # 确保dim能被num_heads整除
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 简化的相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # 获取相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化相对位置偏置表
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
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
    """Swin Transformer Block，参考官方实现"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # 如果窗口大小大于输入分辨率，我们不分割窗口
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Identity()  # 简化版本不使用drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer，参考官方实现"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.reduction(x)
        x = self.norm(x)
        
        return x


class SwinBackbone(nn.Module):
    """
    简化的Swin Transformer Backbone
    保持与现有YOLOv8 Backbone相同的接口
    """
    def __init__(self, w_factor, r_factor_unused, n_factor):
        super().__init__()
        
        # 基础配置
        base_embed_dim = 96
        depths = [2, 2, 6, 2]  # 每个stage的block数量
        base_num_heads = [3, 6, 12, 24]
        window_size = 7
        
        # 根据w_factor调整通道数，确保能被注意力头数整除
        self.embed_dim = max(32, int(base_embed_dim * w_factor))
        # 确保embed_dim能被最小的注意力头数整除
        min_heads = min(base_num_heads)
        self.embed_dim = ((self.embed_dim + min_heads - 1) // min_heads) * min_heads
        
        # 调整注意力头数以匹配通道数
        num_heads = []
        for base_heads in base_num_heads:
            heads = max(1, int(base_heads * w_factor)) if w_factor >= 0.5 else max(1, base_heads // 2)
            # 确保当前stage的embed_dim能被heads整除
            stage_dim = self.embed_dim * (2 ** len(num_heads))
            while stage_dim % heads != 0 and heads > 1:
                heads -= 1
            num_heads.append(heads)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=640, patch_size=4, in_chans=3, embed_dim=self.embed_dim
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        
        dim = self.embed_dim
        for i in range(4):
            # 当前stage的输入分辨率
            input_resolution = (640 // (4 * 2**i), 640 // (4 * 2**i))
            
            # Swin Transformer blocks with alternating shift
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,  # 交替使用移位窗口
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Patch merging (except for the last stage)
            if i < 3:
                patch_merging = PatchMerging(
                    input_resolution=input_resolution,
                    dim=dim,
                    norm_layer=nn.LayerNorm
                )
                self.patch_merging_layers.append(patch_merging)
                dim = dim * 2
        
        # 输出通道适配层 - 适配到YOLOv8期望的通道数
        target_channels = [
            max(1, int(128 * w_factor)),   # feat0 (P2) - 对应原backbone的layer 1输出
            max(1, int(256 * w_factor)),   # feat1 (P3) - 对应原backbone的layer 4输出  
            max(1, int(512 * w_factor)),   # feat2 (P4) - 对应原backbone的layer 6输出
            max(1, int(1024 * w_factor)),  # feat3 (P5) - 对应原backbone的layer 9输出
        ]
        
        # 当前Swin各stage的输出通道数
        swin_channels = [
            self.embed_dim,      # Stage 0
            self.embed_dim * 2,  # Stage 1  
            self.embed_dim * 4,  # Stage 2
            self.embed_dim * 8,  # Stage 3
        ]
        
        # 通道适配层
        self.channel_adapters = nn.ModuleList([
            Conv(swin_ch, target_ch, k=1, s=1) 
            for swin_ch, target_ch in zip(swin_channels, target_channels)
        ])

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape (与原backbone保持一致):
            feat0: (B, 128 * w, 160, 160)  
            feat1: (B, 256 * w, 80, 80)   
            feat2: (B, 512 * w, 40, 40)   
            feat3: (B, 1024 * w, 20, 20)  
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        features = []
        current_H, current_W = H // 4, W // 4  # 初始分辨率 160x160
        
        for i in range(4):
            # Swin Transformer blocks
            for block in self.stages[i]:
                x = block(x)
            
            # 将token序列转换回特征图格式
            feat = x.transpose(1, 2).view(B, -1, current_H, current_W)
            
            # 通道适配
            feat = self.channel_adapters[i](feat)
            features.append(feat)
            
            # Patch merging (除了最后一个stage)
            if i < 3:
                x = self.patch_merging_layers[i](x)
                current_H, current_W = current_H // 2, current_W // 2
        
        return features[0], features[1], features[2], features[3] 