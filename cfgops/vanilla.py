import os
from config import mconfig


def mcfg(tags):
    mcfg = mconfig.ModelConfig()
    # projectRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # pretrainedFile = os.path.join(projectRootDir, "resources/pretrained/backbone", "backbone_{}.pth".format(mcfg.phase))
    # mcfg.pretrainedBackboneUrl = "file://{}".format(pretrainedFile)

    mcfg.phase = "nano" # DO NOT MODIFY
    mcfg.trainSplitName = "train" # DO NOT MODIFY
    mcfg.validationSplitName = "validation" # DO NOT MODIFY
    mcfg.testSplitName = "test" # DO NOT MODIFY

    # data setup
    mcfg.imageDir = "/home/v5/Mars/mar20/images"
    mcfg.annotationDir = "/home/v5/Mars/mar20/annotations"
    mcfg.classList = ["A{}".format(x) for x in range(1, 21)] # DO NOT MODIFY
    mcfg.subsetMap = { # DO NOT MODIFY
        "train": "/home/v5/Mars/mar20/splits/v5/train.txt",
        "validation": "/home/v5/Mars/mar20/splits/v5/validation.txt",
        "test": "/home/v5/Mars/mar20/splits/v5/test.txt",
        "small": "/home/v5/Mars/mar20/splits/v5/small.txt",
    }

    mcfg.paintImages = True

    # Data Augmentation configuration - control aug.py parameters
    # Base augmentations (originally default enabled) are configured here.
    # Additional augmentations like erase and crop are enabled by the 'aug' tag.
    mcfg.augmentation = {
        'jitter': 0.3,
        'rescalef': (0.25, 2), # Rescale factor range
        'flipProb': 0.5, # Probability of horizontal flip
        'huef': 0.1,
        'satf': 0.7,
        'valf': 0.4,
        'use_flip': True, # Toggle: Random horizontal flip (Default Enabled)
        'use_hsv': True,  # Toggle: HSV color adjustment (Default Enabled)

        # Additional augmentations (Default Disabled, enabled by 'aug' tag)
        'eraseProb': 0.0, # Probability of applying random erasing
        'eraseArea': (0.02, 0.4), # Area ratio of erased region (min, max)
        'use_erase': False, # Toggle: Random erasing
        'use_crop': False, # Toggle: Random cropping
        'cropArea': (0.5, 1.0), # Crop area ratio range
        
        # MixUp augmentation
        'use_mixup': False, # Toggle: MixUp augmentation
        'mixupProb': 0.0, # Probability of applying MixUp
        'mixupAlpha': 1.5, # MixUp alpha parameter
        
        # Mosaic augmentation
        'use_mosaic': False, # Toggle: Mosaic augmentation
        'mosaicProb': 0.0, # Probability of applying Mosaic
    }

    # YOLOv8官方预训练权重配置 - 使用现有的pretrainedBackboneUrl机制
    if "pretrained" in tags:
        mcfg.pretrainedBackboneUrl = "file://yolov8n_converted_backbone.pth"

    # EMA configuration - optional training enhancement technique
    if "ema" in tags:
        mcfg.use_ema = True
        mcfg.ema_decay = 0.9999  # recommended decay rate
        mcfg.ema_tau = 2000      # recommended tau parameter

    # Early Stopping configuration - optional training enhancement technique
    if "earlystop" in tags:
        mcfg.use_early_stopping = True
        mcfg.early_stopping_patience = 15  # Stop if no improvement for 15 epochs
        mcfg.early_stopping_min_delta = 0.001  # Minimum change to qualify as improvement
        mcfg.early_stopping_monitor = 'val_loss'  # Monitor validation loss
        mcfg.early_stopping_mode = 'min'  # Stop when val_loss stops decreasing

    # Classification Loss Weight Enhancement - boost classification learning
    if "clsboost" in tags:
        # 增强分类损失权重，从默认的0.5提升到2.0
        mcfg.lossWeights = (7.5, 2.0, 1.5)  # box, cls(enhanced), dfl
        
    # Strong Classification Loss Weight Enhancement - aggressive boost
    if "clsboost2x" in tags:
        # 更强的分类损失权重增强，提升到4.0
        mcfg.lossWeights = (7.5, 4.0, 1.5)  # box, cls(strong enhanced), dfl
        
    # Balanced Loss Weights - equal emphasis on all components
    if "balancedloss" in tags:
        # 平衡的损失权重配置
        mcfg.lossWeights = (2.0, 2.0, 2.0)  # box, cls, dfl (all equal)

    # Dynamic Classification Weight Adjustment - gradually increase cls weight
    if "dynclsweight" in tags:
        mcfg.use_dynamic_cls_weight = True
        mcfg.cls_weight_schedule = 'linear'  # Linear increase
        mcfg.cls_weight_start = 2.0  # Start with default weight
        mcfg.cls_weight_end = 4.0    # End with enhanced weight
        mcfg.cls_weight_warmup_epochs = 100  # Gradual increase over 100 epochs
        
    # Cosine Dynamic Classification Weight - smooth cosine increase
    if "cosclsweight" in tags:
        mcfg.use_dynamic_cls_weight = True
        mcfg.cls_weight_schedule = 'cosine'  # Cosine increase
        mcfg.cls_weight_start = 2.0
        mcfg.cls_weight_end = 4.0
        mcfg.cls_weight_warmup_epochs = 80

    # Focal Loss Weight Adjustment - adaptive weight based on classification performance
    if "focalweight" in tags:
        mcfg.use_focal_cls_weight = True
        mcfg.focal_alpha = 2.0  # Controls the rate of down-weighting easy examples
        mcfg.focal_gamma = 2.0  # Focusing parameter
        mcfg.min_cls_weight = 0.1  # Minimum weight when classification is very good
        mcfg.max_cls_weight = 8.0  # Maximum weight when classification is poor
        
    # Strong Focal Loss Weight - more aggressive focal adjustment
    if "focalweight2x" in tags:
        mcfg.use_focal_cls_weight = True
        mcfg.focal_alpha = 3.0  # Higher alpha for stronger effect
        mcfg.focal_gamma = 3.0  # Higher gamma for more focus on hard examples
        mcfg.min_cls_weight = 0.05
        mcfg.max_cls_weight = 10.0

    # Varifocal Loss - quality-aware focal loss for dense object detection
    if "varifocal" in tags:
        mcfg.use_varifocal_loss = True
        mcfg.varifocal_gamma = 2.0  # Focusing parameter
        mcfg.varifocal_alpha = 0.75  # Balancing factor
        
    # Strong Varifocal Loss - more aggressive varifocal adjustment
    if "varifocal2x" in tags:
        mcfg.use_varifocal_loss = True
        mcfg.varifocal_gamma = 3.0  # Higher gamma for stronger focusing
        mcfg.varifocal_alpha = 0.8   # Higher alpha for better balance
        
    # Standard Focal Loss - classic focal loss implementation
    if "focalloss" in tags:
        mcfg.use_focal_loss = True
        mcfg.focal_gamma = 1.5  # Standard gamma
        mcfg.focal_alpha = 0.25  # Standard alpha
        
    # Strong Focal Loss - more aggressive focal loss
    if "focalloss2x" in tags:
        mcfg.use_focal_loss = True
        mcfg.focal_gamma = 2.5  # Higher gamma for stronger effect
        mcfg.focal_alpha = 0.3   # Higher alpha

    # Enable additional data augmentation features via 'aug' tag
    if "aug" in tags:
        mcfg.augmentation['use_erase'] = True # Enable erase
        mcfg.augmentation['eraseProb'] = 0.5 # Set erase probability (can override default 0.0 if needed, though 0.0 is the 'off' state)
        mcfg.augmentation['use_crop'] = True # Enable crop
        mcfg.augmentation['cropArea'] = (0.5, 1.0) # Set crop area
        mcfg.augmentation['use_mixup'] = True # Enable Mixup
        mcfg.augmentation['mixupProb'] = 1.0 # Set Mixup probability
        mcfg.augmentation['mixupAlpha'] = 1.5 # Set Mixup alpha
        mcfg.augmentation['use_mosaic'] = True # Enable Mosaic
        mcfg.augmentation['mosaicProb'] = 0.5 # Set Mosaic probability (50% chance)

    # Enhanced NMS Configuration - Class-Specific Non-Maximum Suppression
    if "enhancednms" in tags:
        mcfg.use_enhanced_nms = True  # Enable class-specific NMS
    else:
        mcfg.use_enhanced_nms = False  # Use traditional class-agnostic NMS by default

    # AdamW Optimizer Configuration - Modern adaptive optimizer
    if "adamw" in tags:
        mcfg.optimizerType = "AdamW"
        mcfg.baseLearningRate = 1e-3  # Lower learning rate for AdamW (typical range: 1e-4 to 1e-3)
        mcfg.optimizerWeightDecay = 1e-2  # Higher weight decay for AdamW (typical: 1e-2 to 1e-1)
        mcfg.optimizerBetas = (0.9, 0.999)  # Standard AdamW beta parameters
        mcfg.optimizerEps = 1e-8  # Numerical stability parameter

    # Swin Transformer Backbone Configuration - Vision Transformer backbone
    if "swin" in tags:
        mcfg.backbone_type = "swin"
        # Swin Transformer通常需要更小的学习率
        if mcfg.optimizerType == "SGD":
            mcfg.baseLearningRate = 5e-3  # 降低SGD学习率
        elif mcfg.optimizerType == "AdamW":
            mcfg.baseLearningRate = 5e-4  # 降低AdamW学习率

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]

    if "long" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 20000
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]

    if "fast" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 50
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]

    if "teacher" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]
        mcfg.trainSelectedClasses = ["A{}".format(x) for x in range(1, 11)] # DO NOT MODIFY

    if "distillation" in tags:
        mcfg.modelName = "distillation"
        mcfg.checkpointModelFile = "/home/v5/Mars/v5/vanilla.nano.teacher/__cache__/best_weights.pth"
        mcfg.teacherModelFile = "/home/v5/Mars/v5/vanilla.nano.teacher/__cache__/best_weights.pth"
        mcfg.distilLossWeights = (1.0, 0.5, 0.2)
        mcfg.maxEpoch = 300
        mcfg.backboneFreezeEpochs = [x for x in range(0, 25)]
        mcfg.epochValidation = False # DO NOT MODIFY
        mcfg.trainSplitName = "small" # DO NOT MODIFY
        mcfg.teacherClassIndexes = [x for x in range(0, 10)] # DO NOT MODIFY
        
        # Added based on ResponseLoss requirements
        mcfg.nc = len(mcfg.classList) # Student total classes
        mcfg.teacher_nc = 10          # Teacher's original number of classes (for distillation target)
        mcfg.regMax = 16              # Common value for YOLO/DFL, adjust if necessary
        mcfg.teacher_head_total_classes = len(mcfg.classList) # Actual total classes in teacher's head output

    return mcfg
