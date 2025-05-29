import torch
import torch.nn as nn
from model.base.components import Conv, C2f # Assuming Concat will be torch.cat


class Neck(nn.Module):
    """
    YOLOv8 PAFPN Neck.
    Reference: yolov8.yaml 'head' section (excluding Detect layer)
    """
    def __init__(self, w_factor, r_factor_unused, n_factor):
        super().__init__()

        gw = w_factor  # width_multiplier, e.g., 0.25 for nano
        gd = n_factor / 3.0  # depth_multiplier, e.g., 1.0/3.0 for nano

        # Helper to calculate number of Bottleneck repeats in C2f
        def get_repeats(base_repeats):
            # In YAML head, C2f repeats are fixed (e.g., 3), not scaled by depth usually.
            # However, to maintain consistency with Backbone if scaling is desired for Neck C2f:
            # return max(1, round(base_repeats * gd))
            # For YOLOv8 head/neck, Ultralytics official code often uses fixed repeats for C2f in head.
            # Let's use fixed repeats from YAML (typically 3 for each C2f in head).
            # The YAML defines '3' for each C2f in the head for all scales.
            return base_repeats # Use base_repeats directly from YAML if not scaling C2f in Neck by gd
        
        # Re-evaluating get_repeats: The YAML shows '3' for all C2f in head. Let's stick to that for now.
        # If scaling by gd for neck C2f is needed, it can be changed. For now, direct value from YAML.
        # Let's assume yaml's `[-1, 3, C2f, [args]]` means 3 repeats, unscaled by gd for the head C2f blocks.
        # This seems to be the common practice in Ultralytics YOLOv8 models for the head part.

        def get_channels(base_channels):
            return max(1, int(base_channels * gw))

        # Input channels from Backbone (for nano, after width scaling by gw=0.25):
        # feat1_b (P3 from Backbone Layer 4 out): get_channels(256) = 64
        # feat2_b (P4 from Backbone Layer 6 out): get_channels(512) = 128
        # feat3_b (P5 from Backbone Layer 9 out): get_channels(1024) = 256
        ch_b_p3 = get_channels(256)  # P3 output from Backbone (feat1_b)
        ch_b_p4 = get_channels(512)  # P4 output from Backbone (feat2_b)
        ch_b_p5 = get_channels(1024) # P5 output from Backbone (feat3_b)

        # Top-down path
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest") # YAML layer 10 (head index 0)
        # Concat input channels for layer 12: ch_b_p5 (after up1) + ch_b_p4
        ch_concat1 = ch_b_p5 + ch_b_p4 
        self.c2f_p4_fused = C2f(ch_concat1, get_channels(512), n=3, shortcut=False) # YAML layer 12 (head index 2), n=3 from YAML
        ch_p4_fused = get_channels(512)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest") # YAML layer 13 (head index 3)
        # Concat input channels for layer 15: ch_p4_fused (after up2) + ch_b_p3
        ch_concat2 = ch_p4_fused + ch_b_p3
        self.c2f_p3_out = C2f(ch_concat2, get_channels(256), n=3, shortcut=False) # YAML layer 15 (head index 5), n=3 from YAML
        ch_p3_out_channels = get_channels(256)

        # Bottom-up path
        self.down1_conv = Conv(ch_p3_out_channels, get_channels(256), k=3, s=2) # YAML layer 16 (head index 6)
        ch_down1_out = get_channels(256)
        # Concat input channels for layer 18: ch_down1_out + ch_p4_fused (from layer 12)
        ch_concat3 = ch_down1_out + ch_p4_fused
        self.c2f_p4_out = C2f(ch_concat3, get_channels(512), n=3, shortcut=False) # YAML layer 18 (head index 8), n=3 from YAML
        ch_p4_out_channels = get_channels(512)

        self.down2_conv = Conv(ch_p4_out_channels, get_channels(512), k=3, s=2) # YAML layer 19 (head index 9)
        ch_down2_out = get_channels(512)
        # Concat input channels for layer 21: ch_down2_out + ch_b_p5 (original P5 from backbone)
        ch_concat4 = ch_down2_out + ch_b_p5
        self.c2f_p5_out = C2f(ch_concat4, get_channels(1024), n=3, shortcut=False) # YAML layer 21 (head index 11), n=3 from YAML

        # Store the channel of p4_fused for the 'C' output in forward, if needed by docstring.
        # This is mainly to match the signature, YoloModel ignores the first return value.
        self.ch_p4_fused_for_C_output = ch_p4_fused 

    def forward(self, feat1_b, feat2_b, feat3_b):
        """
        Input shape (for nano, w=0.25, r=2):
            feat1_b (P3 from Backbone): (B, 64, 80, 80)
            feat2_b (P4 from Backbone): (B, 128, 40, 40)
            feat3_b (P5 from Backbone): (B, 256, 20, 20)
        Output shape (as expected by YoloModel: _, X, Y, Z for nano):
            C_out (ignored): (B, 128, 40, 40) (p4_fused)
            X_out (P3_neck): (B, 64, 80, 80)
            Y_out (P4_neck): (B, 128, 40, 40)
            Z_out (P5_neck): (B, 256, 20, 20)
        """
        # Top-down path
        # feat3_b is P5 output from backbone (e.g., nano: 256 channels, 20x20)
        p5_up = self.up1(feat3_b)  # Upsamples to 40x40. Channels: 256 (nano)
        # feat2_b is P4 output from backbone (e.g., nano: 128 channels, 40x40)
        p4_concat_in = torch.cat([p5_up, feat2_b], dim=1) # Channels: 256+128=384 (nano)
        p4_fused = self.c2f_p4_fused(p4_concat_in) # Output channels: 128 (nano), 40x40. This is Layer 12 in YAML head. 
                                                # This can be our C_out (ignored by YoloModel)

        p4_up = self.up2(p4_fused) # Upsamples to 80x80. Channels: 128 (nano)
        # feat1_b is P3 output from backbone (e.g., nano: 64 channels, 80x80)
        p3_concat_in = torch.cat([p4_up, feat1_b], dim=1) # Channels: 128+64=192 (nano)
        p3_out_neck = self.c2f_p3_out(p3_concat_in) # Output channels: 64 (nano), 80x80. This is X_out (P3_neck), Layer 15 in YAML head.

        # Bottom-up path
        p3_down = self.down1_conv(p3_out_neck) # Downsamples to 40x40. Channels: 64 (nano). Layer 16 in YAML head.
        p4_concat_in2 = torch.cat([p3_down, p4_fused], dim=1) # Channels: 64+128=192 (nano). Layer 17 in YAML head.
        p4_out_neck = self.c2f_p4_out(p4_concat_in2) # Output channels: 128 (nano), 40x40. This is Y_out (P4_neck), Layer 18 in YAML head.

        p4_down = self.down2_conv(p4_out_neck) # Downsamples to 20x20. Channels: 128 (nano). Layer 19 in YAML head.
        # feat3_b is P5 output from backbone (e.g., nano: 256 channels, 20x20)
        p5_concat_in2 = torch.cat([p4_down, feat3_b], dim=1) # Channels: 128+256=384 (nano). Layer 20 in YAML head.
        p5_out_neck = self.c2f_p5_out(p5_concat_in2) # Output channels: 256 (nano), 20x20. This is Z_out (P5_neck), Layer 21 in YAML head.
        
        return p4_fused, p3_out_neck, p4_out_neck, p5_out_neck
