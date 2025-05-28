import torch
import torch.nn as nn
import math
from .components import Conv, C2f, SPPF

class Backbone(nn.Module):
    """
    YOLOv8 Backbone (CSPDarknet P5 variant)
    Reference: resources/yolov8.md
    """
    def __init__(self, w, r, n): # w: widen_factor, r: ratio_p5, n: deepen_factor
        super().__init__()

        # Helper to calculate number of channels (make divisible by 8)
        def make_c(base_channels, widen_factor, ratio=1.0):
            return math.ceil(base_channels * widen_factor * ratio / 8) * 8

        # Helper to calculate number of bottleneck repetitions
        def make_n_bottlenecks(base_depth, deepen_factor):
            return max(round(base_depth * deepen_factor), 1)

        # Input channels for the image
        img_channels = 3

        # --- Stem Layer (Layer 0) ---
        # Output: 320x320x(64*w)
        c_stem_out = make_c(64, w)
        self.stem = Conv(img_channels, c_stem_out, k=3, s=2, p=1)

        # --- Stage 1 (Layers 1, 2) --- output feat0 (P2 features)
        # Layer 1: ConvModule. Output: 160x160x(128*w)
        # Layer 2: CSPLayer_2Conv. Output: 160x160x(128*w)
        c_s1_conv_in = c_stem_out
        c_s1_conv_out = make_c(128, w)
        self.s1_conv = Conv(c_s1_conv_in, c_s1_conv_out, k=3, s=2, p=1)
        n_s1_csp = make_n_bottlenecks(3, n) # n=3xd in yolov8.md
        self.s1_csp = C2f(c_s1_conv_out, c_s1_conv_out, n=n_s1_csp, shortcut=True, e=0.5)

        # --- Stage 2 (Layers 3, 4) --- output feat1 (P3 features for Neck)
        # Layer 3: ConvModule. Output: 80x80x(256*w)
        # Layer 4: CSPLayer_2Conv. Output: 80x80x(256*w)
        c_s2_conv_in = c_s1_conv_out
        c_s2_conv_out = make_c(256, w)
        self.s2_conv = Conv(c_s2_conv_in, c_s2_conv_out, k=3, s=2, p=1)
        n_s2_csp = make_n_bottlenecks(6, n) # n=6xd in yolov8.md
        self.s2_csp = C2f(c_s2_conv_out, c_s2_conv_out, n=n_s2_csp, shortcut=True, e=0.5)

        # --- Stage 3 (Layers 5, 6) --- output feat2 (P4 features for Neck)
        # Layer 5: ConvModule. Output: 40x40x(512*w)
        # Layer 6: CSPLayer_2Conv. Output: 40x40x(512*w)
        c_s3_conv_in = c_s2_conv_out
        c_s3_conv_out = make_c(512, w)
        self.s3_conv = Conv(c_s3_conv_in, c_s3_conv_out, k=3, s=2, p=1)
        n_s3_csp = make_n_bottlenecks(6, n) # n=6xd in yolov8.md
        self.s3_csp = C2f(c_s3_conv_out, c_s3_conv_out, n=n_s3_csp, shortcut=True, e=0.5)

        # --- Stage 4 (Layers 7, 8, 9) --- output feat3 (P5 features for Neck)
        # Layer 7: ConvModule. Output: 20x20x(512*w*r)
        # Layer 8: CSPLayer_2Conv. Output: 20x20x(512*w*r)
        # Layer 9: SPPF. Output: 20x20x(512*w*r)
        c_s4_conv_in = c_s3_conv_out
        c_s4_conv_out = make_c(512, w, ratio=r) # r is the ratio_p5 from init
        self.s4_conv = Conv(c_s4_conv_in, c_s4_conv_out, k=3, s=2, p=1)
        n_s4_csp = make_n_bottlenecks(3, n) # n=3xd in yolov8.md
        self.s4_csp = C2f(c_s4_conv_out, c_s4_conv_out, n=n_s4_csp, shortcut=True, e=0.5)
        self.sppf = SPPF(c_s4_conv_out, c_s4_conv_out, k=5)

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape (for a P5 backbone configuration):
            feat0: (B, 128*w, H/4, W/4)   # P2 features (e.g., 160x160 for 640 input)
            feat1: (B, 256*w, H/8, W/8)   # P3 features (e.g., 80x80 for 640 input) - To Neck
            feat2: (B, 512*w, H/16, W/16) # P4 features (e.g., 40x40 for 640 input) - To Neck
            feat3: (B, 512*w*r, H/32, W/32) # P5 features (e.g., 20x20 for 640 input) - To Neck
        """
        x = self.stem(x)        # Input to S1 (P1 level features)

        x_s1_conv = self.s1_conv(x)     # Output of S1 Conv (P2 level)
        feat0 = self.s1_csp(x_s1_conv)  # Output of S1 CSP (P2 features, e.g., 160x160)

        x_s2_conv = self.s2_conv(feat0) # Output of S2 Conv (P3 level)
        feat1 = self.s2_csp(x_s2_conv)  # Output of S2 CSP (P3 features, e.g., 80x80)

        x_s3_conv = self.s3_conv(feat1) # Output of S3 Conv (P4 level)
        feat2 = self.s3_csp(x_s3_conv)  # Output of S3 CSP (P4 features, e.g., 40x40)

        x_s4_conv = self.s4_conv(feat2) # Output of S4 Conv (P5 level)
        x_s4_csp_out = self.s4_csp(x_s4_conv) # Output of S4 CSP
        feat3 = self.sppf(x_s4_csp_out) # Output of S4 SPPF (P5 features, e.g., 20x20)

        return feat0, feat1, feat2, feat3
