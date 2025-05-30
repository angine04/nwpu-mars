# engine/trainer/ema.py
import torch
from copy import deepcopy

class ModelEMA:
    """EMA (Exponential Moving Average) model weight management"""
    def __init__(self, model, decay=0.9999, tau=2000):
        # Parameter validation
        if not (0 <= decay <= 1):
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        if tau <= 0:
            raise ValueError(f"EMA tau must be positive, got {tau}")
        
        self.decay = decay
        self.tau = tau
        self.updates = 0
        
        # Ensure EMA model is on the same device
        self.ema = deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        """Update EMA model with current model parameters"""
        self.updates += 1
        
        # Device compatibility check
        if hasattr(model, 'outputDevice'):
            device = model.outputDevice()
        else:
            device = next(model.parameters()).device
        
        if self.ema.parameters().__next__().device != device:
            self.ema = self.ema.to(device)
        
        # Dynamic decay calculation
        decay = self.decay * (1 - torch.exp(-torch.tensor(self.updates / self.tau, dtype=torch.float32)))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def copy_attr(self, model):
        """Copy non-parameter attributes from original model (if needed)"""
        # Copy some important model attributes
        for attr in ['stride', 'nc', 'names', 'anchors']:
            if hasattr(model, attr):
                setattr(self.ema, attr, getattr(model, attr))
    
    def clone_model_attr(self, model):
        """Clone non-parameter attributes from model"""
        for k, v in model.__dict__.items():
            if not k.startswith('_') and k not in ['training']:
                try:
                    setattr(self.ema, k, v)
                except (AttributeError, TypeError):
                    pass
    
    def __repr__(self):
        return f'ModelEMA(decay={self.decay}, tau={self.tau}, updates={self.updates})'