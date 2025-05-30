# engine/trainer/ema.py
import torch
import math
from copy import deepcopy

class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA to
            decay (float): decay factor for exponential moving average
            tau (int): number of updates over which decay ramps up from 0 to decay
            updates (int): number of EMA updates already performed
        """
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = self.de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Update EMA attributes"""
        self.copy_attr(self.ema, model, include, exclude)

    @staticmethod
    def de_parallel(model):
        """De-parallelize a model: returns single-GPU model if model is of type DP or DDP"""
        return model.module if hasattr(model, 'module') else model

    @staticmethod  
    def copy_attr(a, b, include=(), exclude=()):
        """Copy attributes from b to a, options to only include [...] and to exclude [...]"""
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v) 