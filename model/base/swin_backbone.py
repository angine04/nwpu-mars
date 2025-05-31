import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging
from model.base.components import SPPF  # 保留原始SPPF模块

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=640, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class SwinStage(nn.Module):
    """ A basic Swin Transformer stage """
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                norm_layer=nn.LayerNorm
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SwinBackbone(nn.Module):
    """ 
    Swin-Transformer Backbone for YOLOv8
    Maintains same feature map dimensions as original backbone
    """
    def __init__(self, w_factor, r_factor_unused, n_factor, img_size=640):
        super().__init__()
        
        gw = w_factor  # width_multiplier
        gd = n_factor / 3.0  # depth_multiplier
        
        def get_channels(base_channels):
            return max(1, int(base_channels * gw))
        
        def get_repeats(base_repeats):
            return max(1, round(base_repeats * gd))
        
        # Output channels to match original backbone
        ch_p2 = get_channels(128)  # feat0 (160x160)
        ch_p3 = get_channels(256)  # feat1 (80x80)
        ch_p4 = get_channels(512)  # feat2 (40x40)
        ch_p5 = get_channels(1024) # feat3 (20x20)
        
        # Swin-Tiny like configuration (adjustable)
        embed_dim = int(96 * gw)
        depths = [int(2 * gd), int(2 * gd), int(6 * gd), int(2 * gd)]
        num_heads = [3, 6, 12, 24]  # Adjust based on embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim
        )
        
        # Stages with patch merging
        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0])
        self.merge1 = PatchMerging(embed_dim, 2*embed_dim)
        
        self.stage2 = SwinStage(2*embed_dim, depths[1], num_heads[1])
        self.merge2 = PatchMerging(2*embed_dim, 4*embed_dim)
        
        self.stage3 = SwinStage(4*embed_dim, depths[2], num_heads[2])
        self.merge3 = PatchMerging(4*embed_dim, 8*embed_dim)
        
        self.stage4 = SwinStage(8*embed_dim, depths[3], num_heads[3])
        
        # Channel adapters to match original backbone output channels
        self.adapter1 = nn.Conv2d(embed_dim, ch_p2, 1) if embed_dim != ch_p2 else nn.Identity()
        self.adapter2 = nn.Conv2d(2*embed_dim, ch_p3, 1) if 2*embed_dim != ch_p3 else nn.Identity()
        self.adapter3 = nn.Conv2d(4*embed_dim, ch_p4, 1) if 4*embed_dim != ch_p4 else nn.Identity()
        self.adapter4 = nn.Conv2d(8*embed_dim, ch_p5, 1) if 8*embed_dim != ch_p5 else nn.Identity()
        
        # SPPF at the end to match original backbone
        self.sppf = SPPF(ch_p5, ch_p5, k=5)
        
        # Store output channels
        self.channels = [ch_p2, ch_p3, ch_p4, ch_p5]

    def forward(self, x):
        """ Forward pass that maintains same feature map dimensions as original backbone """
        # Initial patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/4, W/4)
        
        # Stage 1 (160x160)
        x = self.stage1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat0 = self.adapter1(x)
        
        # Stage 2 (80x80)
        x = self.merge1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.stage2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat1 = self.adapter2(x)
        
        # Stage 3 (40x40)
        x = self.merge2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.stage3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat2 = self.adapter3(x)
        
        # Stage 4 (20x20)
        x = self.merge3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.stage4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.adapter4(x)
        feat3 = self.sppf(x)
        
        return feat0, feat1, feat2, feat3