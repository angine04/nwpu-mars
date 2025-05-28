import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override # this could be removed since Python 3.12

from train.loss import DetectionLoss # Assuming DetectionLoss is in train.loss


class CWDLoss(nn.Module):
    """Channel-wise Distillation Loss"""
    def __init__(self, device, tau=1.0):
        super().__init__()
        self.device = device
        self.tau = tau

    def forward(self, s_feats, t_feats):
        loss = torch.tensor(0.0, device=self.device)
        if not isinstance(s_feats, (list, tuple)) or not isinstance(t_feats, (list, tuple)):
            # Handle cases where features might not be lists/tuples (e.g. single tensor)
            # This part might need adjustment based on actual model output structure
            s_feats = [s_feats] if torch.is_tensor(s_feats) else []
            t_feats = [t_feats] if torch.is_tensor(t_feats) else []

        if len(s_feats) != len(t_feats):
            # print(f"Warning: CWDLoss student features count ({len(s_feats)}) != teacher features count ({len(t_feats)}). Skipping CWD.")
            return loss

        for s_f, t_f in zip(s_feats, t_feats):
            if s_f.shape[1] != t_f.shape[1]:
                # print(f"Warning: CWDLoss channel mismatch ({s_f.shape[1]} vs {t_f.shape[1]}). Skipping this pair.")
                continue
            
            b, c, h, w = s_f.shape
            s_f_reshaped = s_f.view(b, c, h * w)
            t_f_reshaped = t_f.view(b, c, h * w)

            log_softmax_s = F.log_softmax(s_f_reshaped / self.tau, dim=2)
            softmax_t = F.softmax(t_f_reshaped / self.tau, dim=2)
            
            # Sum over spatial dimension, then mean over channels and batch
            kl_div_pair = F.kl_div(log_softmax_s, softmax_t, reduction='none').sum(dim=2).mean()
            loss += kl_div_pair
        
        return loss


class ResponseLoss(nn.Module):
    """Distillation Loss for Head Responses (Classification Part)"""
    def __init__(self, device, nc, reg_max, teacher_class_indexes=None, tau=1.0):
        super().__init__()
        self.device = device
        self.nc = nc
        self.reg_max = reg_max
        self.teacher_class_indexes = teacher_class_indexes
        if self.teacher_class_indexes is not None:
            self.teacher_class_indexes = torch.tensor(teacher_class_indexes, device=device, dtype=torch.long)
        self.tau = tau

    def forward(self, s_resp_tuple, t_resp_tuple):
        loss = torch.tensor(0.0, device=self.device)

        for s_r, t_r in zip(s_resp_tuple, t_resp_tuple):
            # s_r, t_r shape: (B, regMax * 4 + nc, H, W)
            s_cls = s_r.split((self.reg_max * 4, self.nc), dim=1)[1]
            t_cls = t_r.split((self.reg_max * 4, self.nc), dim=1)[1]

            if self.teacher_class_indexes is not None and t_cls.shape[1] != self.nc:
                t_cls = torch.index_select(t_cls, 1, self.teacher_class_indexes)
            
            if s_cls.shape[1] != t_cls.shape[1]: # Should match after teacher_class_indexes adjustment
                # print(f"Warning: ResponseLoss class count mismatch ({s_cls.shape[1]} vs {t_cls.shape[1]}). Skipping this pair.")
                continue

            b_s, nc_s, h_s, w_s = s_cls.shape
            s_cls_flat = s_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc_s)
            
            b_t, nc_t, h_t, w_t = t_cls.shape
            t_cls_flat = t_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc_t)

            log_softmax_s = F.log_softmax(s_cls_flat / self.tau, dim=1)
            softmax_t = F.softmax(t_cls_flat / self.tau, dim=1)

            kl_div_pair = F.kl_div(log_softmax_s, softmax_t, reduction='batchmean')
            loss += kl_div_pair
        
        return loss


class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False # Unused currently, kept from original
        
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(device=mcfg.device, tau=getattr(mcfg, 'cwd_tau', 1.0))
        self.respLoss = ResponseLoss(device=mcfg.device, 
                                     nc=mcfg.nc, 
                                     reg_max=mcfg.regMax, 
                                     teacher_class_indexes=getattr(mcfg, 'teacherClassIndexes', None),
                                     tau=getattr(mcfg, 'resp_tau', 1.0))

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds is a tuple (student_preds_and_feats, teacher_preds_and_feats)
        Each element of rawPreds is typically a tuple itself:
          ( (head_p3, head_p4, head_p5), feat1, feat2, ... )
        So, spreds would be: ( (s_head_p3, ...), s_feat1, ...)
        
        The slicing spreds[:3] and spreds[3:] assumes this structure where
        the first element is the tuple of head outputs and the rest are features.
        Let's adjust if model output is a flat tuple of tensors.
        If model(x) returns (head_outs_tuple, feats_tuple):
            s_outputs, t_outputs = rawPreds[0], rawPreds[1]
            sresponse, sfeats = s_outputs[0], s_outputs[1]
            tresponse, tfeats = t_outputs[0], t_outputs[1]
        The original slicing spreds[:3] implies spreds is a flat list/tuple of tensors
        where first 3 are head outputs and rest are features.
        Example from original comment:
            spreds = (head_p3, head_p4, head_p5, feat_p2, feat_p3, feat_p4, feat_p5_bb, neck_p3, neck_p4, neck_p5_nk)
            sresponse = (head_p3, head_p4, head_p5)
            sfeats    = (feat_p2, ...)
        This interpretation will be kept.
        """
        spreds_tuple = rawPreds[0] # Student's full output tuple
        tpreds_tuple = rawPreds[1] # Teacher's full output tuple

        # Assuming the model's forward pass for distillation returns a flat tuple of tensors:
        # (head_out1, head_out2, head_out3, feat1, feat2, ...)
        sresponse, sfeats = spreds_tuple[:3], list(spreds_tuple[3:])
        tresponse, tfeats = tpreds_tuple[:3], list(tpreds_tuple[3:])

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        
        # Original detection loss for student
        loss[0] = self.detectionLoss(sresponse, batch) # batch contains ground truth targets
        
        # CWD feature distillation loss
        loss[1] = self.cwdLoss(sfeats, tfeats)
        
        # Response-based distillation loss (on head outputs)
        loss[2] = self.respLoss(sresponse, tresponse)

        # Apply overall weights for each distillation component
        loss[0] *= self.mcfg.distilLossWeights[0]  # original loss weight
        loss[1] *= self.mcfg.distilLossWeights[1]  # cwd distillation weight
        loss[2] *= self.mcfg.distilLossWeights[2]  # response distillation weight

        return loss.sum()
