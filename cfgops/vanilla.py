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
    mcfg.augmentation = {
        'jitter': 0.3,
        'rescalef': (0.25, 2),
        'flipProb': 0.5,
        'huef': 0.1,
        'satf': 0.7,
        'valf': 0.4,
    }

    # YOLOv8官方预训练权重配置 - 使用现有的pretrainedBackboneUrl机制
    if "pretrained" in tags:
        mcfg.pretrainedBackboneUrl = "file://yolov8n_converted_backbone.pth"

    # EMA configuration - optional training enhancement technique
    if "ema" in tags:
        mcfg.use_ema = True
        mcfg.ema_decay = 0.9999  # recommended decay rate
        mcfg.ema_tau = 2000      # recommended tau parameter

    if "full" in tags:
        mcfg.modelName = "base"
        mcfg.maxEpoch = 200
        mcfg.backboneFreezeEpochs = [x for x in range(0, 100)]

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
