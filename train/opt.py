import torch
import torch.nn as nn
import torch.optim as optim


class MarsOptimizerFactory(object):
    @staticmethod
    def initOptimizer(mcfg, model):
        match mcfg.optimizerType:
            case "SGD":
                return MarsOptimizerFactory.initSgdOptimizer(mcfg, model)
            case "AdamW":
                return MarsOptimizerFactory.initAdamWOptimizer(mcfg, model)
            case other:
                raise ValueError("Invalid optimizer type: {}".format(mcfg.optimizerType))

    @staticmethod
    def getModelParameterGroups(model):
        weights, bnWeights, bias = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    bias.append(param)
                elif isinstance(module, bn):  # batch norm layer weights (no decay)
                    bnWeights.append(param)
                else:  # normal weights (with decay)
                    weights.append(param)
        return weights, bnWeights, bias

    @staticmethod
    def initSgdOptimizer(mcfg, model):
        weights, bnWeights, bias = MarsOptimizerFactory.getModelParameterGroups(model)
        opt = optim.SGD(
            bnWeights,
            lr=mcfg.baseLearningRate,
            momentum=mcfg.optimizerMomentum,
            nesterov=True,
        )
        opt.add_param_group({"params": weights, "weight_decay": mcfg.optimizerWeightDecay})
        opt.add_param_group({"params": bias})
        return opt

    @staticmethod
    def initAdamWOptimizer(mcfg, model):
        """
        Initialize AdamW optimizer with proper parameter grouping.
        AdamW is particularly effective for transformer-based models and modern deep learning.
        """
        weights, bnWeights, bias = MarsOptimizerFactory.getModelParameterGroups(model)
        
        # AdamW optimizer with parameter groups
        # Group 1: BatchNorm weights (no weight decay)
        opt = optim.AdamW(
            bnWeights,
            lr=mcfg.baseLearningRate,
            betas=getattr(mcfg, 'optimizerBetas', (0.9, 0.999)),
            eps=getattr(mcfg, 'optimizerEps', 1e-8),
            weight_decay=0.0,  # No weight decay for BN weights
        )
        
        # Group 2: Regular weights (with weight decay)
        opt.add_param_group({
            "params": weights, 
            "weight_decay": mcfg.optimizerWeightDecay
        })
        
        # Group 3: Bias parameters (no weight decay)
        opt.add_param_group({
            "params": bias,
            "weight_decay": 0.0
        })
        
        return opt
