# engine/trainer/ema.py
import torch
from copy import deepcopy

class ModelEMA:
    """EMA（指数移动平均）模型权重管理"""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = deepcopy(model).eval()  # 初始化EMA模型（冻结状态）
        self.decay = lambda x: decay * (1 - torch.exp(-torch.tensor(x / tau)))  # 动态衰减
        self.updates = 0  # 更新计数器

    def update(self, model):
        """更新EMA参数"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            # print(f"\nEMA Update #{self.updates}: decay={d:.6f}")
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(d).add_((1 - d) * model_p.detach())