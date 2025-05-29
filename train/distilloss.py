import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override # this could be removed since Python 3.12
from train.loss import DetectionLoss # Assuming DetectionLoss is in train.loss

class CWDLoss(nn.Module):
    """
    Channel-wise Distillation for Dense Prediction.
    This is a simplified implementation focusing on the core idea.
    Assumes student and teacher feature maps for CWD have the same number of channels
    or that channel adaptation is handled before calling this loss.
    """
    def __init__(self, spatial_weight_type='mean_abs', channel_loss_type='l2'):
        super().__init__()
        self.spatial_weight_type = spatial_weight_type
        self.channel_loss_type = channel_loss_type

    def _calculate_spatial_weight(self, t_feat):
        b, c, h, w = t_feat.shape
        if self.spatial_weight_type == 'mean_abs':
            spatial_map = torch.mean(torch.abs(t_feat), dim=1, keepdim=True)
        elif self.spatial_weight_type == 'sum_sq':
            spatial_map = torch.sum(t_feat**2, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown spatial_weight_type: {self.spatial_weight_type}")
        spatial_map_flat = spatial_map.view(b, -1)
        spatial_map_norm = spatial_map_flat / (torch.sum(spatial_map_flat, dim=1, keepdim=True) + 1e-7)
        return spatial_map_norm.view(b, 1, h, w)

    @override
    def __call__(self, sfeats, tfeats):
        total_cwd_loss = 0.0
        num_distilled_layers = 0
        if not sfeats: 
            return torch.tensor(0.0).to(next(self.parameters()).device if next(self.parameters(), None) is not None else "cpu")
            
        for s_f, t_f in zip(sfeats, tfeats):
            if s_f.shape[1] != t_f.shape[1]:
                print(f"Warning: CWD layer {num_distilled_layers} skipped due to channel mismatch: S({s_f.shape[1]}) vs T({t_f.shape[1]}).")
                continue
            if s_f.shape[2:] != t_f.shape[2:]:
                print(f"Warning: CWD layer {num_distilled_layers} skipped due to spatial mismatch: S({s_f.shape[2:]}) vs T({t_f.shape[2:]}).")
                continue

            spatial_weights = self._calculate_spatial_weight(t_f)
            term_inside_sum_c = (s_f - t_f)**2 if self.channel_loss_type == 'l2' else torch.abs(s_f - t_f)
            sum_over_c = torch.sum(term_inside_sum_c, dim=1, keepdim=True)
            weighted_by_spatial_map = sum_over_c * spatial_weights
            current_loss = torch.mean(torch.sum(weighted_by_spatial_map, dim=(2,3)))
            total_cwd_loss += current_loss
            num_distilled_layers += 1
        
        return total_cwd_loss / num_distilled_layers if num_distilled_layers > 0 else torch.tensor(0.0).to(sfeats[0].device)

class ResponseLoss(nn.Module):
    """
    Distillation loss based on model's response (classification logits).
    """
    def __init__(self, mcfg_student_nc, mcfg_teacher_nc_for_distill, mcfg_teacher_head_total_classes, mcfg_teacher_class_indexes, mcfg_reg_max, mcfg_distill_temp=1.0, mcfg_response_loss_type='kldiv'):
        super().__init__()
        self.student_total_classes = mcfg_student_nc
        self.teacher_nc_for_distill = mcfg_teacher_nc_for_distill # Number of old classes to distill
        self.teacher_head_total_classes = mcfg_teacher_head_total_classes # Actual total classes in teacher's head
        self.teacher_class_indexes = mcfg_teacher_class_indexes # Indexes to select old classes
        self.reg_max = mcfg_reg_max
        self.temperature = mcfg_distill_temp
        self.loss_type = mcfg_response_loss_type

        if self.loss_type == 'kldiv':
            self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        elif self.loss_type == 'mse':
            self.mse_loss = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss_type for ResponseLoss: {self.loss_type}")

    def _extract_cls_logits(self, pred_tensor, num_classes_in_pred_head):
        # num_classes_in_pred_head is the total number of class channels in the prediction tensor's head
        expected_channels = self.reg_max * 4 + num_classes_in_pred_head
        if pred_tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Prediction tensor channel mismatch. Expected {expected_channels} (regMax*4 + num_classes_in_head = {self.reg_max*4} + {num_classes_in_pred_head}), "
                f"got {pred_tensor.shape[1]}. Ensure regMax and num_classes_in_head are correctly set for this prediction tensor."
            )
        return pred_tensor[:, self.reg_max * 4:, :, :]

    @override
    def __call__(self, sresponse, tresponse):
        total_response_loss = 0.0
        num_heads = 0
        if not sresponse:
             return torch.tensor(0.0).to(next(self.parameters()).device if next(self.parameters(), None) is not None else "cpu")
             
        for s_pred, t_pred in zip(sresponse, tresponse):
            # Extract all classification logits from student based on its head configuration
            s_cls_logits_all_student = self._extract_cls_logits(s_pred, self.student_total_classes)
            # Select the logits corresponding to old classes from the student's perspective
            s_cls_logits_for_old_classes = s_cls_logits_all_student[:, self.teacher_class_indexes, :, :]

            # Extract all classification logits from teacher based on its head configuration
            t_cls_logits_all_teacher = self._extract_cls_logits(t_pred, self.teacher_head_total_classes)
            # Select the logits corresponding to old classes from the teacher's perspective (these are the distillation targets)
            t_cls_logits_distill_target = t_cls_logits_all_teacher[:, self.teacher_class_indexes, :, :]
            
            # Validate that the number of selected old classes match
            if s_cls_logits_for_old_classes.shape[1] != t_cls_logits_distill_target.shape[1]:
                raise ValueError(
                    f"Mismatch in number of distilled classes. Student has {s_cls_logits_for_old_classes.shape[1]} (selected via teacher_class_indexes), "
                    f"Teacher target has {t_cls_logits_distill_target.shape[1]} (selected via teacher_class_indexes). "
                    f"This usually means teacher_class_indexes is inconsistent with teacher_nc_for_distill ({self.teacher_nc_for_distill})."
                )
            # Further check if the number of selected classes matches the configured teacher_nc_for_distill
            if t_cls_logits_distill_target.shape[1] != self.teacher_nc_for_distill:
                raise ValueError(
                    f"Number of classes in teacher's distillation target ({t_cls_logits_distill_target.shape[1]}) "
                    f"does not match configured teacher_nc_for_distill ({self.teacher_nc_for_distill}). Check teacher_class_indexes."
                )

            if self.loss_type == 'kldiv':
                log_s_probs = F.log_softmax(s_cls_logits_for_old_classes / self.temperature, dim=1)
                log_t_probs = F.log_softmax(t_cls_logits_distill_target / self.temperature, dim=1) # Use teacher's selected old class logits
                b, nc_old, h, w = log_s_probs.shape # nc_old should be self.teacher_nc_for_distill
                current_loss = self.kl_loss(
                    log_s_probs.permute(0, 2, 3, 1).reshape(b * h * w, nc_old),
                    log_t_probs.permute(0, 2, 3, 1).reshape(b * h * w, nc_old)
                )
            elif self.loss_type == 'mse':
                current_loss = self.mse_loss(
                    s_cls_logits_for_old_classes / self.temperature,
                    t_cls_logits_distill_target / self.temperature # Use teacher's selected old class logits
                )
            total_response_loss += current_loss
            num_heads += 1
            
        return total_response_loss / num_heads if num_heads > 0 else torch.tensor(0.0).to(sresponse[0].device)

class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model): # model is the student model
        self.mcfg = mcfg
        self.histMode = False 
        
        self.detectionLoss = DetectionLoss(mcfg, model)
        
        self.cwdLoss = CWDLoss(
            spatial_weight_type=getattr(mcfg, 'cwd_spatial_weight_type', 'mean_abs'),
            channel_loss_type=getattr(mcfg, 'cwd_channel_loss_type', 'l2')
        )
        
        required_attrs_resp = ['nc', 'teacher_nc', 'teacher_head_total_classes', 'teacherClassIndexes', 'regMax']
        if not all(hasattr(mcfg, attr) for attr in required_attrs_resp):
            missing_attrs = [attr for attr in required_attrs_resp if not hasattr(mcfg, attr)]
            raise AttributeError(
                f"mcfg is missing one or more required attributes for ResponseLoss: {missing_attrs}. "
                "Needed: nc (student_nc), teacher_nc (old classes for distill), "
                "teacher_head_total_classes (total classes in teacher head), teacherClassIndexes, regMax"
            )
        self.respLoss = ResponseLoss(
            mcfg_student_nc=mcfg.nc,
            mcfg_teacher_nc_for_distill=mcfg.teacher_nc, # Number of old classes teacher provides supervision for
            mcfg_teacher_head_total_classes=mcfg.teacher_head_total_classes, # Actual total classes in teacher's head output
            mcfg_teacher_class_indexes=mcfg.teacherClassIndexes,
            mcfg_reg_max=mcfg.regMax,
            mcfg_distill_temp=getattr(mcfg, 'distill_temp', 1.0),
            mcfg_response_loss_type=getattr(mcfg, 'response_loss_type', 'kldiv')
        )
        
        if not hasattr(mcfg, 'distilLossWeights') or len(mcfg.distilLossWeights) != 3:
             raise AttributeError("mcfg.distilLossWeights must be a list/tuple of 3 float values.")

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80), # student_nc for student, teacher_head_total_classes for teacher
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
