import math
import torch
from misc.log import log


class LossWeightScheduler:
    """
    动态损失权重调度器，用于在训练过程中调整损失权重
    特别适用于分类损失权重的动态调整
    """
    
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.use_dynamic_cls_weight = getattr(mcfg, 'use_dynamic_cls_weight', False)
        
        if self.use_dynamic_cls_weight:
            self.schedule_type = getattr(mcfg, 'cls_weight_schedule', 'linear')
            self.start_weight = getattr(mcfg, 'cls_weight_start', 0.5)
            self.end_weight = getattr(mcfg, 'cls_weight_end', 2.0)
            self.warmup_epochs = getattr(mcfg, 'cls_weight_warmup_epochs', 50)
            
            # 保存原始权重配置
            self.original_weights = list(mcfg.lossWeights)
            
            log.green(f"Dynamic classification weight scheduler initialized:")
            log.green(f"  Schedule: {self.schedule_type}")
            log.green(f"  Weight range: {self.start_weight} -> {self.end_weight}")
            log.green(f"  Warmup epochs: {self.warmup_epochs}")
    
    def get_cls_weight(self, epoch):
        """根据当前epoch计算分类损失权重"""
        if not self.use_dynamic_cls_weight:
            return self.original_weights[1]  # 返回原始分类权重
        
        if epoch >= self.warmup_epochs:
            return self.end_weight
        
        # 计算权重调整进度
        progress = epoch / self.warmup_epochs
        
        if self.schedule_type == 'linear':
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress
        elif self.schedule_type == 'cosine':
            weight = self.start_weight + (self.end_weight - self.start_weight) * (1 - math.cos(progress * math.pi)) / 2
        elif self.schedule_type == 'step':
            # 阶梯式调整，在warmup_epochs的一半时跳跃到目标权重
            if progress < 0.5:
                weight = self.start_weight
            else:
                weight = self.end_weight
        else:
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress
        
        return weight
    
    def update_loss_weights(self, epoch):
        """更新损失权重配置"""
        if not self.use_dynamic_cls_weight:
            return self.mcfg.lossWeights
        
        new_cls_weight = self.get_cls_weight(epoch)
        
        # 更新分类损失权重（索引1）
        new_weights = list(self.original_weights)
        new_weights[1] = new_cls_weight
        
        # 更新配置
        self.mcfg.lossWeights = tuple(new_weights)
        
        return self.mcfg.lossWeights
    
    def log_current_weights(self, epoch):
        """记录当前的损失权重"""
        if self.use_dynamic_cls_weight:
            weights = self.mcfg.lossWeights
            log.inf(f"Epoch {epoch + 1}: Loss weights = (box: {weights[0]:.3f}, cls: {weights[1]:.3f}, dfl: {weights[2]:.3f})")


class FocalLossWeightScheduler(LossWeightScheduler):
    """
    基于Focal Loss思想的权重调度器
    根据分类准确率动态调整权重
    """
    
    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.use_focal_weight = getattr(mcfg, 'use_focal_cls_weight', False)
        
        if self.use_focal_weight:
            self.focal_alpha = getattr(mcfg, 'focal_alpha', 2.0)
            self.focal_gamma = getattr(mcfg, 'focal_gamma', 2.0)
            self.min_cls_weight = getattr(mcfg, 'min_cls_weight', 0.1)
            self.max_cls_weight = getattr(mcfg, 'max_cls_weight', 5.0)
            
            # 保存原始权重配置
            self.original_weights = list(mcfg.lossWeights)
            
            # 用于跟踪分类损失历史
            self.cls_loss_history = []
            self.window_size = 5  # 使用最近5个epoch的平均值
            
            log.green(f"Focal loss weight scheduler initialized:")
            log.green(f"  Alpha: {self.focal_alpha}, Gamma: {self.focal_gamma}")
            log.green(f"  Weight range: {self.min_cls_weight} - {self.max_cls_weight}")
    
    def update_focal_weight_by_loss(self, cls_loss, total_loss):
        """根据分类损失占比动态调整权重"""
        if not self.use_focal_weight:
            return self.mcfg.lossWeights
        
        # 计算分类损失占总损失的比例
        cls_loss_ratio = cls_loss / max(total_loss, 1e-8)
        
        # 添加到历史记录
        self.cls_loss_history.append(cls_loss_ratio)
        if len(self.cls_loss_history) > self.window_size:
            self.cls_loss_history.pop(0)
        
        # 计算平均分类损失比例
        avg_cls_ratio = sum(self.cls_loss_history) / len(self.cls_loss_history)
        
        # Focal weight计算：损失比例越高，说明分类越困难，权重应该越高
        # 使用sigmoid函数将比例映射到权重范围
        normalized_ratio = min(max(avg_cls_ratio * 10, 0), 1)  # 将比例放大并限制在[0,1]
        focal_weight = self.focal_alpha * (normalized_ratio ** self.focal_gamma)
        focal_weight = max(self.min_cls_weight, min(self.max_cls_weight, focal_weight))
        
        # 更新分类损失权重
        new_weights = list(self.original_weights)
        new_weights[1] = focal_weight
        
        self.mcfg.lossWeights = tuple(new_weights)
        
        return self.mcfg.lossWeights
    
    def update_focal_weight(self, cls_accuracy):
        """根据分类准确率更新权重"""
        if not self.use_focal_weight:
            return self.mcfg.lossWeights
        
        # Focal weight: 准确率越低，权重越高
        focal_weight = self.focal_alpha * (1 - cls_accuracy) ** self.focal_gamma
        focal_weight = max(self.min_cls_weight, min(self.max_cls_weight, focal_weight))
        
        # 更新分类损失权重
        new_weights = list(self.original_weights)
        new_weights[1] = focal_weight
        
        self.mcfg.lossWeights = tuple(new_weights)
        
        return self.mcfg.lossWeights
    
    def update_loss_weights(self, epoch, cls_loss=None, total_loss=None):
        """重写父类方法，支持基于损失的focal调整"""
        if not self.use_focal_weight:
            return super().update_loss_weights(epoch)
        
        # 如果提供了损失信息，使用基于损失的调整
        if cls_loss is not None and total_loss is not None:
            return self.update_focal_weight_by_loss(cls_loss, total_loss)
        
        # 否则使用默认的epoch-based调整（如果同时启用了动态调整）
        if self.use_dynamic_cls_weight:
            return super().update_loss_weights(epoch)
        
        # 纯focal模式下，保持当前权重
        return self.mcfg.lossWeights
    
    def log_current_weights(self, epoch):
        """记录当前的损失权重和focal信息"""
        if self.use_focal_weight:
            weights = self.mcfg.lossWeights
            if self.cls_loss_history:
                avg_cls_ratio = sum(self.cls_loss_history) / len(self.cls_loss_history)
                log.inf(f"Epoch {epoch + 1}: Loss weights = (box: {weights[0]:.3f}, cls: {weights[1]:.3f}, dfl: {weights[2]:.3f}), "
                       f"avg_cls_ratio: {avg_cls_ratio:.4f}")
            else:
                log.inf(f"Epoch {epoch + 1}: Loss weights = (box: {weights[0]:.3f}, cls: {weights[1]:.3f}, dfl: {weights[2]:.3f})")
        elif self.use_dynamic_cls_weight:
            super().log_current_weights(epoch) 