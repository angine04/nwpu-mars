# engine/trainer/ema.py
import torch
from copy import deepcopy

class ModelEMA:
    """EMA（指数移动平均）模型权重管理"""
    def __init__(self, model, decay=0.9999, tau=2000):
        # 🔧 参数验证
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        if tau <= 0:
            raise ValueError(f"EMA tau must be positive, got {tau}")
        
        # 初始化EMA模型（深拷贝并设为eval模式）
        self.ema = deepcopy(model).eval()
        
        # 🔧 确保EMA模型在同一设备上
        if hasattr(model, 'outputDevice'):
            device = model.outputDevice()
        else:
            # fallback: 使用模型参数的设备
            device = next(model.parameters()).device
        self.ema = self.ema.to(device)
        
        # 动态衰减函数
        self.decay_fn = lambda x: decay * (1 - torch.exp(-torch.tensor(x / tau, dtype=torch.float32)))
        self.updates = 0
        self.decay = decay
        self.tau = tau
        
        # 冻结EMA模型的所有参数
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        """更新EMA参数"""
        # 🔧 设备一致性检查
        model_device = next(model.parameters()).device
        ema_device = next(self.ema.parameters()).device
        if model_device != ema_device:
            raise RuntimeError(f"Model and EMA model on different devices: {model_device} vs {ema_device}")
        
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates).item()  # 确保是标量
            
            # 更新EMA参数
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.mul_(d).add_(model_param.detach(), alpha=1.0 - d)
    
    def copy_attr(self, model):
        """从原模型复制非参数属性（如果需要）"""
        # 复制一些重要的模型属性
        for attr in ['stride', 'nc', 'names', 'anchors']:
            if hasattr(model, attr):
                setattr(self.ema, attr, getattr(model, attr))
    
    def clone_model_attr(self, model):
        """克隆模型的非参数属性"""
        for k, v in model.__dict__.items():
            if not k.startswith('_') and k not in ['training']:
                try:
                    setattr(self.ema, k, v)
                except (AttributeError, TypeError):
                    pass
    
    def __repr__(self):
        return f'ModelEMA(decay={self.decay}, tau={self.tau}, updates={self.updates})'