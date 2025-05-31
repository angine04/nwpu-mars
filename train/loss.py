import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from misc.bbox import bboxDecode, iou, bbox2dist
from train.tal import TaskAlignedAssigner


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with torch.amp.autocast('cuda', enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


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
        
        # 选择分类损失函数
        self.use_varifocal = getattr(mcfg, 'use_varifocal_loss', False)
        self.use_focal = getattr(mcfg, 'use_focal_loss', False)
        
        if self.use_varifocal:
            self.cls_loss = VarifocalLoss(
                gamma=getattr(mcfg, 'varifocal_gamma', 2.0),
                alpha=getattr(mcfg, 'varifocal_alpha', 0.75)
            )
        elif self.use_focal:
            self.cls_loss = FocalLoss(
                gamma=getattr(mcfg, 'focal_gamma', 1.5),
                alpha=getattr(mcfg, 'focal_alpha', 0.25)
            )
        else:
            self.cls_loss = nn.BCEWithLogitsLoss(reduction="none")
            
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

        # 使用model中的anchor points和stride tensor
        anchorPoints = self.model.anchorPoints
        strideTensor = self.model.anchorStrides
        
        # 解码预测框
        predBboxes = bboxDecode(anchorPoints, predBoxDistribution, self.model.proj, xywh=False)  # xyxy, (b, h*w, 4)
        
        # 使用Task-Aligned Assigner分配目标
        _, targetBboxes, targetScores, fgMask, _ = self.assigner(
            predClassScores.detach().sigmoid(),
            (predBboxes.detach() * strideTensor).type(gtBboxes.dtype),
            anchorPoints * strideTensor,
            gtLabels,
            gtBboxes,
            gtMask,
        )
        
        targetScoresSum = max(targetScores.sum(), 1)
        
        # 分类损失
        if self.use_varifocal:
            # Varifocal Loss需要IoU质量分数作为gt_score
            if fgMask.sum():
                # 计算正样本的IoU质量分数 - 使用高效的逐元素IoU计算
                pred_boxes_fg = predBboxes[fgMask]  # (num_positive, 4)
                target_boxes_fg = targetBboxes[fgMask]  # (num_positive, 4)
                
                # 直接计算逐元素IoU，避免广播
                # 将两个张量reshape为相同形状以确保逐元素计算
                iou_scores = self._compute_elementwise_iou(pred_boxes_fg, target_boxes_fg)
                
                # 为正样本设置IoU质量分数，负样本保持0
                quality_scores = targetScores.clone()
                quality_scores[fgMask] = quality_scores[fgMask] * iou_scores.unsqueeze(-1)
                loss[1] = self.cls_loss(predClassScores, quality_scores.to(predClassScores.dtype), targetScores.to(predClassScores.dtype)) / targetScoresSum
            else:
                # 没有正样本时，使用标准BCE
                loss[1] = F.binary_cross_entropy_with_logits(predClassScores, targetScores.to(predClassScores.dtype), reduction="none").sum() / targetScoresSum
        elif self.use_focal:
            # Focal Loss使用标准输入
            loss[1] = self.cls_loss(predClassScores, targetScores.to(predClassScores.dtype)) / targetScoresSum
        else:
            # 标准BCE Loss
            loss[1] = self.cls_loss(predClassScores, targetScores.to(predClassScores.dtype)).sum() / targetScoresSum
        
        # 边界框损失
        if fgMask.sum():
            targetBboxes /= strideTensor
            loss[0], loss[2] = self.bboxLoss(
                predBoxDistribution, predBboxes, anchorPoints, targetBboxes, targetScores, targetScoresSum, fgMask
            )

        loss[0] *= self.mcfg.lossWeights[0]  # box
        loss[1] *= self.mcfg.lossWeights[1]  # cls
        loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return loss.sum()
    
    def get_loss_components(self, preds, targets):
        """
        返回分离的损失组件，用于Focal Loss权重调度器
        Returns: (total_loss, box_loss, cls_loss, dfl_loss)
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

        # 使用model中的anchor points和stride tensor
        anchorPoints = self.model.anchorPoints
        strideTensor = self.model.anchorStrides
        
        # 解码预测框
        predBboxes = bboxDecode(anchorPoints, predBoxDistribution, self.model.proj, xywh=False)  # xyxy, (b, h*w, 4)
        
        # 使用Task-Aligned Assigner分配目标
        _, targetBboxes, targetScores, fgMask, _ = self.assigner(
            predClassScores.detach().sigmoid(),
            (predBboxes.detach() * strideTensor).type(gtBboxes.dtype),
            anchorPoints * strideTensor,
            gtLabels,
            gtBboxes,
            gtMask,
        )
        
        targetScoresSum = max(targetScores.sum(), 1)
        
        # 分类损失（未加权）
        if self.use_varifocal:
            # Varifocal Loss需要IoU质量分数作为gt_score
            if fgMask.sum():
                # 计算正样本的IoU质量分数 - 使用高效的逐元素IoU计算
                pred_boxes_fg = predBboxes[fgMask]  # (num_positive, 4)
                target_boxes_fg = targetBboxes[fgMask]  # (num_positive, 4)
                
                # 直接计算逐元素IoU，避免广播
                # 将两个张量reshape为相同形状以确保逐元素计算
                iou_scores = self._compute_elementwise_iou(pred_boxes_fg, target_boxes_fg)
                
                # 为正样本设置IoU质量分数，负样本保持0
                quality_scores = targetScores.clone()
                quality_scores[fgMask] = quality_scores[fgMask] * iou_scores.unsqueeze(-1)
                cls_loss_raw = self.cls_loss(predClassScores, quality_scores.to(predClassScores.dtype), targetScores.to(predClassScores.dtype)) / targetScoresSum
            else:
                # 没有正样本时，使用标准BCE
                cls_loss_raw = F.binary_cross_entropy_with_logits(predClassScores, targetScores.to(predClassScores.dtype), reduction="none").sum() / targetScoresSum
        elif self.use_focal:
            # Focal Loss使用标准输入
            cls_loss_raw = self.cls_loss(predClassScores, targetScores.to(predClassScores.dtype)) / targetScoresSum
        else:
            # 标准BCE Loss
            cls_loss_raw = self.cls_loss(predClassScores, targetScores.to(predClassScores.dtype)).sum() / targetScoresSum
        
        loss[1] = cls_loss_raw
        
        # 边界框损失（未加权）
        if fgMask.sum():
            targetBboxes /= strideTensor
            loss[0], loss[2] = self.bboxLoss(
                predBoxDistribution, predBboxes, anchorPoints, targetBboxes, targetScores, targetScoresSum, fgMask
            )

        # 应用权重
        weighted_loss = loss.clone()
        weighted_loss[0] *= self.mcfg.lossWeights[0]  # box
        weighted_loss[1] *= self.mcfg.lossWeights[1]  # cls
        weighted_loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return weighted_loss.sum(), loss[0].item(), loss[1].item(), loss[2].item()

    def _compute_elementwise_iou(self, boxes1, boxes2):
        """
        计算两组框的逐元素IoU（对应位置的IoU）。
        
        Args:
            boxes1 (torch.Tensor): 形状为 (N, 4) 的张量，表示预测框
            boxes2 (torch.Tensor): 形状为 (N, 4) 的张量，表示目标框
            
        Returns:
            torch.Tensor: 形状为 (N,) 的张量，包含对应位置的IoU值
        """
        assert boxes1.shape == boxes2.shape, f"boxes1 shape {boxes1.shape} != boxes2 shape {boxes2.shape}"
        
        # 直接计算逐元素IoU，使用xyxy格式
        # 获取边界框坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.chunk(4, -1)
        
        # 计算交集
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # 交集面积
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # 计算各自面积
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        # 并集面积
        union_area = area1 + area2 - inter_area + 1e-7
        
        # IoU
        iou_scores = (inter_area / union_area).squeeze(-1).detach().clamp(0)
        
        return iou_scores
