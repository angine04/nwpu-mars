import torch
import torch.nn as nn


from .components import Conv, C2f


def make_c(base_channels, widen_factor, ratio=1.0, divisor=8):
    """Calculates number of channels, ensuring it's divisible by divisor."""
    return int(max(round(base_channels * widen_factor * ratio), 1) * divisor / divisor) * divisor


def make_n(base_n, deepen_factor):
    """Calculates number of bottleneck repetitions."""
    return max(round(base_n * deepen_factor), 1)


class Neck(nn.Module):
    """
    YOLOv8 PAFPN Neck.
    Reference: resources/yolov8.md (derived from yolov8.jpg)
    Takes P3, P4, P5 features from Backbone and outputs fused features for the Head.
    """
    def __init__(self, widen_factor, ratio_p5, deepen_factor):
        super().__init__()

        n_csp = make_n(3, deepen_factor) # Number of bottlenecks in C2f layers for the neck

        # Channels from backbone outputs
        # feat1 (P3): c_bb_p3 = make_c(256, widen_factor)
        # feat2 (P4): c_bb_p4 = make_c(512, widen_factor)
        # feat3 (P5): c_bb_p5 = make_c(512, widen_factor, ratio_p5)

        # Upsample layer (reused)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-Down Path
        # Layer 12: CSPLayer_2Conv (P4_fused)
        # Input: Upsampled P5 (c_bb_p5) + Backbone P4 (c_bb_p4)
        c1_td1 = make_c(512, widen_factor, ratio_p5) + make_c(512, widen_factor)
        c2_td1 = make_c(512, widen_factor)
        self.csp_td1 = C2f(c1_td1, c2_td1, n=n_csp, shortcut=True, e=0.5)

        # Layer 15: CSPLayer_2Conv (P3_fused) -> Output for Head (smallest stride)
        # Input: Upsampled P4_fused (c2_td1) + Backbone P3 (c_bb_p3)
        c1_td2 = c2_td1 + make_c(256, widen_factor)
        self.c_p3_out = make_c(256, widen_factor)
        self.csp_td2 = C2f(c1_td2, self.c_p3_out, n=n_csp, shortcut=True, e=0.5)

        # Bottom-Up Path
        # Layer 16: ConvModule for downsampling P3_fused
        self.conv_bu1 = Conv(self.c_p3_out, self.c_p3_out, k=3, s=2)

        # Layer 18: CSPLayer_2Conv (P4_fused_bottomup) -> Output for Head (medium stride)
        # Input: Downsampled P3_fused (self.c_p3_out) + P4_fused (c2_td1)
        c1_bu1 = self.c_p3_out + c2_td1
        self.c_p4_out = make_c(512, widen_factor)
        self.csp_bu1 = C2f(c1_bu1, self.c_p4_out, n=n_csp, shortcut=True, e=0.5)

        # Layer 19: ConvModule for downsampling P4_fused_bottomup
        self.conv_bu2 = Conv(self.c_p4_out, self.c_p4_out, k=3, s=2)

        # Layer 21: CSPLayer_2Conv (P5_fused_bottomup) -> Output for Head (largest stride)
        # Input: Downsampled P4_fused_bottomup (self.c_p4_out) + Backbone P5 (c_bb_p5)
        c1_bu2 = self.c_p4_out + make_c(512, widen_factor, ratio_p5)
        c2_bu2 = make_c(512, widen_factor, ratio_p5)
        self.csp_bu2 = C2f(c1_bu2, c2_bu2, n=n_csp, shortcut=True, e=0.5)

    def forward(self, feat1_p3, feat2_p4, feat3_p5):
        """
        Args:
            feat1_p3: Backbone P3 features (e.g., B, C1, H/8, W/8)
            feat2_p4: Backbone P4 features (e.g., B, C2, H/16, W/16)
            feat3_p5: Backbone P5 features (e.g., B, C3, H/32, W/32)

        Returns:
            (p3_out, p4_out, p5_out): Tuple of fused feature maps for the Head.
                p3_out: Features from Neck Layer 15 (e.g., B, 256*w, H/8, W/8)
                p4_out: Features from Neck Layer 18 (e.g., B, 512*w, H/16, W/16)
                p5_out: Features from Neck Layer 21 (e.g., B, 512*w*r, H/32, W/32)
        """
        # Top-Down Path
        # P5 from backbone (feat3_p5) -> P4_fused
        up_p5 = self.upsample(feat3_p5) # Upsample P5
        cat_p5_p4 = torch.cat([up_p5, feat2_p4], dim=1) # Concat with Backbone P4
        p4_fused = self.csp_td1(cat_p5_p4) # Layer 12 output

        # P4_fused -> P3_fused (p3_out)
        up_p4_fused = self.upsample(p4_fused) # Upsample P4_fused
        cat_p4f_p3 = torch.cat([up_p4_fused, feat1_p3], dim=1) # Concat with Backbone P3
        p3_out = self.csp_td2(cat_p4f_p3) # Layer 15 output (to Head)

        # Bottom-Up Path
        # P3_fused (p3_out) -> P4_fused_bottomup (p4_out)
        down_p3_out = self.conv_bu1(p3_out) # Downsample P3_fused
        cat_dp3_p4f = torch.cat([down_p3_out, p4_fused], dim=1) # Concat with P4_fused (from top-down)
        p4_out = self.csp_bu1(cat_dp3_p4f) # Layer 18 output (to Head)

        # P4_fused_bottomup (p4_out) -> P5_fused_bottomup (p5_out)
        down_p4_out = self.conv_bu2(p4_out) # Downsample P4_fused_bottomup
        cat_dp4_p5 = torch.cat([down_p4_out, feat3_p5], dim=1) # Concat with Backbone P5
        p5_out = self.csp_bu2(cat_dp4_p5) # Layer 21 output (to Head)

        return p3_out, p4_out, p5_out
