import torch.nn as nn
from model.base.components import Conv, C2f, SPPF


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg and yolov8.yaml
    """
    def __init__(self, w_factor, r_factor_unused, n_factor): # w_factor, r_factor_unused, n_factor directly from YoloModelPhaseSetup
        super().__init__()

        gw = w_factor  # width_multiplier, e.g., 0.25 for nano
        gd = n_factor / 3.0  # depth_multiplier, e.g., 1.0/3.0 for nano (n_factor is 1 for nano)

        # Helper to calculate number of Bottleneck repeats in C2f
        def get_repeats(base_repeats):
            return max(1, round(base_repeats * gd))

        # Helper to calculate channel numbers
        def get_channels(base_channels):
            return max(1, int(base_channels * gw))

        # Base channels from yolov8n.yaml backbone
        ch_in = 3
        ch_p1_conv = get_channels(64)
        ch_p2_conv = get_channels(128) # Output of this is feat0 for docstring
        ch_p2_c2f = get_channels(128)
        ch_p3_conv = get_channels(256)
        ch_p3_c2f = get_channels(256)  # Output of this is feat1 for Neck
        ch_p4_conv = get_channels(512)
        ch_p4_c2f = get_channels(512)  # Output of this is feat2 for Neck
        ch_p5_conv = get_channels(1024)
        ch_p5_c2f = get_channels(1024) # Output of C2f before SPPF
        ch_sppf_out = get_channels(1024) # Output of this is feat3 for Neck

        # Number of repeats for C2f blocks from yolov8n.yaml backbone
        n_p2_c2f = get_repeats(3)
        n_p3_c2f = get_repeats(6)
        n_p4_c2f = get_repeats(6)
        n_p5_c2f = get_repeats(3)


        self.layers = nn.ModuleList([
            Conv(ch_in, ch_p1_conv, k=3, s=2),               # 0-P1/2 (layer 0 in yaml)
            Conv(ch_p1_conv, ch_p2_conv, k=3, s=2),          # 1-P2/4 (layer 1 in yaml) / feat0_docstring
            C2f(ch_p2_conv, ch_p2_c2f, n=n_p2_c2f, shortcut=True), # (layer 2 in yaml)
            Conv(ch_p2_c2f, ch_p3_conv, k=3, s=2),           # 3-P3/8 (layer 3 in yaml)
            C2f(ch_p3_conv, ch_p3_c2f, n=n_p3_c2f, shortcut=True), # (layer 4 in yaml) / feat1_docstring
            Conv(ch_p3_c2f, ch_p4_conv, k=3, s=2),           # 5-P4/16 (layer 5 in yaml)
            C2f(ch_p4_conv, ch_p4_c2f, n=n_p4_c2f, shortcut=True), # (layer 6 in yaml) / feat2_docstring
            Conv(ch_p4_c2f, ch_p5_conv, k=3, s=2),           # 7-P5/32 (layer 7 in yaml)
            C2f(ch_p5_conv, ch_p5_c2f, n=n_p5_c2f, shortcut=True), # (layer 8 in yaml)
            SPPF(ch_p5_c2f, ch_sppf_out, k=5)                # (layer 9 in yaml) / feat3_docstring
        ])

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape (for YoloModel, based on its usage `_, feat1, feat2, feat3 = self.backbone.forward(x)`):
            out_L1 (feat0_docstring): (B, 128 * w, 160, 160) -> Channel: ch_p2_conv
            out_L4 (feat1_docstring): (B, 256 * w, 80, 80)   -> Channel: ch_p3_c2f
            out_L6 (feat2_docstring): (B, 512 * w, 40, 40)   -> Channel: ch_p4_c2f
            out_L9 (feat3_docstring): (B, 1024 * w, 20, 20)  -> Channel: ch_sppf_out
        The docstring in the original Backbone class had feat3 as 512*w*r, which is 1024*w.
        """
        out_L0 = self.layers[0](x)
        out_L1 = self.layers[1](out_L0) # feat0_docstring
        out_L2 = self.layers[2](out_L1)
        out_L3 = self.layers[3](out_L2)
        out_L4 = self.layers[4](out_L3) # feat1_docstring
        out_L5 = self.layers[5](out_L4)
        out_L6 = self.layers[6](out_L5) # feat2_docstring
        out_L7 = self.layers[7](out_L6)
        out_L8 = self.layers[8](out_L7)
        out_L9 = self.layers[9](out_L8) # feat3_docstring

        return out_L1, out_L4, out_L6, out_L9
