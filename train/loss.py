
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.bbox import bboxDecode, iou, bbox2dist, dist2bbox
from train.tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iouv = iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iouv) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss(object):
    def __init__(self, mcfg, model):
        self.model = model
        self.mcfg= mcfg
        self.layerStrides = model.layerStrides
        self.assigner = TaskAlignedAssigner(topk=self.mcfg.talTopk, num_classes=self.mcfg.nc, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bboxLoss = BboxLoss(self.mcfg.regMax).to(self.mcfg.device)

    def preprocess(self, targets, batchSize, scaleTensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batchSize, 0, ne - 1, device=self.mcfg.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batchSize, counts.max(), ne - 1, device=self.mcfg.device)
            for j in range(batchSize):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scaleTensor)
        return out

    def __call__(self, preds, targets):
        """
        preds shape:
            preds[0]: (B, regMax * 4 + nc, 80, 80)
            preds[1]: (B, regMax * 4 + nc, 40, 40)
            preds[2]: (B, regMax * 4 + nc, 20, 20)
        targets shape:
            (?, 6)
        """
        loss = torch.zeros(3, device=self.mcfg.device)  # box, cls, dfl

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4

        # predictioin preprocess
        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split((self.mcfg.regMax * 4, self.mcfg.nc), 1)
        predBoxDistribution = predBoxDistribution.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, regMax * 4)
        predClassScores = predClassScores.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, nc)

        # ground truth preprocess
        targets = self.preprocess(targets.to(self.mcfg.device), batchSize, scaleTensor=self.model.scaleTensor) # (batchSize, maxCount, 5)
        gtLabels, gtBboxes = targets.split((1, 4), 2)  # cls=(batchSize, maxCount, 1), xyxy=(batchSize, maxCount, 4)
        gtMask = gtBboxes.sum(2, keepdim=True).gt_(0.0)

        # Generate anchor points and stride tensor
        anchor_points_list = []
        stride_tensor_list = []
        for i, stride_val in enumerate(self.layerStrides):
            h, w = preds[i].shape[-2:]
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=self.mcfg.device, dtype=torch.float32),
                                            torch.arange(w, device=self.mcfg.device, dtype=torch.float32), indexing='ij')
            # cell centers (x, y)
            level_anchor_points = torch.stack((grid_x + 0.5, grid_y + 0.5), dim=-1).view(-1, 2)
            anchor_points_list.append(level_anchor_points * stride_val)
            stride_tensor_list.append(torch.full((h * w, 1), stride_val, dtype=torch.float32, device=self.mcfg.device))

        anchor_points = torch.cat(anchor_points_list, dim=0)  # (total_anchors, 2)
        stride_tensor = torch.cat(stride_tensor_list, dim=0) # (total_anchors, 1)

        # Decode predicted bounding boxes
        # predBoxDistribution is (batchSize, total_anchors, 4 * regMax)
        pred_dist_rs = predBoxDistribution.view(batchSize, -1, 4, self.mcfg.regMax)
        pred_dist_sm = F.softmax(pred_dist_rs, dim=-1)
        # Integral of distribution (expectation)
        integ_arange = torch.arange(self.mcfg.regMax, device=self.mcfg.device, dtype=torch.float32)
        pred_offsets_grid_units = (pred_dist_sm * integ_arange).sum(dim=-1) # (batchSize, total_anchors, 4) in ltrb format (grid cell units)

        # Scale offsets by stride to get absolute distances
        pred_offsets_abs = pred_offsets_grid_units * stride_tensor.unsqueeze(0) # (B, total_anchors, 4)

        # Decode to xyxy boxes: bboxDecode(anchor_centers_abs, ltrb_offsets_abs)
        # anchor_points are absolute centers (total_anchors, 2). Need to expand for batch.
        # pred_offsets_abs is already the ltrb distances. Call dist2bbox directly.
        pred_bboxes_decoded_xyxy = dist2bbox(pred_offsets_abs, anchor_points.unsqueeze(0), xywh=False) # (B, total_anchors, 4)

        # Target assignment using TaskAlignedAssigner
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            predClassScores,
            pred_bboxes_decoded_xyxy,
            anchor_points,
            gtLabels,
            gtBboxes,
            gtMask
        )

        # Classification loss (BCEWithLogitsLoss)
        # target_scores from assigner are soft labels (alignment_metric * one_hot_gt_label)
        cls_loss_unreduced = self.bce(predClassScores, target_scores) # (B, total_anchors, nc)
        
        # Normalize by sum of positive target scores
        # fg_mask is (B, total_anchors). target_scores is (B, total_anchors, nc)
        # We want to sum the loss for entries where fg_mask is true.
        # The normalization factor is the sum of target_scores for positive anchors.
        # Ultralytics normalizes by target_scores[fg_mask].sum()
        
        # Sum loss over classes, apply fg_mask, then sum over anchors and batch
        # Normalization factor: sum of all elements in target_scores where fg_mask is true.
        # This means sum(target_scores[fg_mask]) where target_scores[fg_mask] has shape (N_fg, nc)
        # So, effectively sum of all alignment scores for positive matches.
        
        # Simpler: sum the loss where fg_mask is true, normalize by number of positive anchors (fg_mask.sum())
        # Or, as per many TAL implementations, normalize by sum of target_scores[fg_mask]
        
        # Let's use the sum of target_scores over positive anchors for normalization
        # target_scores_sum_for_norm = target_scores[fg_mask].sum()
        # This is slightly different from how BboxLoss expects its target_scores_sum.
        # For cls loss, often normalized by number of positives or sum of target_scores over positives.
        
        # Following a common pattern for TAL: sum cls_loss where fg_mask is true, normalize by sum of target_scores[fg_mask]
        # This means target_scores[fg_mask] is (N_fg, num_classes). Sum of these elements is the normalizer.
        normalizer_cls = target_scores[fg_mask].sum()
        if normalizer_cls > 0:
            loss_cls = (cls_loss_unreduced.sum(dim=-1)[fg_mask]).sum() / normalizer_cls
        else:
            loss_cls = torch.tensor(0.0, device=self.mcfg.device)

        # Bounding box losses (IoU + DFL)
        loss_iou = torch.tensor(0.0, device=self.mcfg.device)
        loss_dfl = torch.tensor(0.0, device=self.mcfg.device)

        if fg_mask.sum() > 0:
            # Select foreground predictions and targets for bbox loss calculation
            pred_dist_fg = predBoxDistribution[fg_mask]                   # (N_fg, 4 * reg_max)
            pred_bboxes_fg = pred_bboxes_decoded_xyxy[fg_mask]            # (N_fg, 4)
            anchor_points_fg = anchor_points.unsqueeze(0).expand(batchSize, -1, -1)[fg_mask] # (N_fg, 2)
            target_bboxes_fg = target_bboxes[fg_mask]                     # (N_fg, 4)

            # For BboxLoss, target_scores should be a per-anchor weight (e.g., alignment score)
            # target_scores from assigner is (B, total_anchors, nc). We need (N_fg, 1).
            # Use the max score of the assigned class for positive anchors.
            bbox_loss_weights_fg = target_scores[fg_mask].max(dim=-1, keepdim=True)[0] # (N_fg, 1)
            bbox_loss_weights_sum = bbox_loss_weights_fg.sum()

            if bbox_loss_weights_sum > 0:
                # The fg_mask argument to BboxLoss is for further filtering if needed, here it's all true for the subset
                loss_iou_val, loss_dfl_val = self.bboxLoss(
                    pred_dist_fg,
                    pred_bboxes_fg,
                    anchor_points_fg,
                    target_bboxes_fg,
                    bbox_loss_weights_fg,    # Weights for each positive sample's loss
                    bbox_loss_weights_sum,   # Normalization factor for the sum of weighted losses
                    torch.ones_like(bbox_loss_weights_fg, dtype=torch.bool).squeeze(-1) # fg_mask for BboxLoss (all true for this subset)
                )
                loss_iou = loss_iou_val
                loss_dfl = loss_dfl_val

        loss[0] = loss_iou
        loss[1] = loss_cls
        loss[2] = loss_dfl

        loss[0] *= self.mcfg.lossWeights[0]  # box (IoU)
        loss[1] *= self.mcfg.lossWeights[1]  # cls
        loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return loss.sum()
