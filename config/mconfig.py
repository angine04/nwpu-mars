import os
import torch
import platform


class ModelConfig(object):
    def __init__(self):
        self.trainer = "base"

        self.user = None
        self.seed = 859
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0") # Default to cuda:0 if multiple GPUs
        else:
            self.device = torch.device("cpu")
        self.cuda = self.device.type == 'cuda' # Keep self.cuda for compatibility if needed elsewhere

        # data setup
        self.imageDir = None
        self.annotationDir = None
        self.classList = None
        self.subsetMap = {}
        self.dcore = 10
        self.suffix = ".jpg"

        # model setup
        self.modelName = "base"
        self.phase = "nano"
        self.backbone_type = "csp"  # 'csp' for CSP Darknet, 'swin' for Swin Transformer
        self.pretrainedBackboneUrl = None
        self.inputShape = (640, 640)
        self.regMax = 16
        self.nc = 20

        # distillation model setup
        self.teacherModelFile = None
        self.distilLossWeights = None
        self.teacherClassIndexes = None

        # train setup
        self.talTopk = 10
        self.lossWeights = (7.5, 0.5, 1.5) # box, cls, dfl
        
        # Dynamic classification loss weight adjustment
        self.use_dynamic_cls_weight = False
        self.cls_weight_schedule = 'linear'  # 'linear', 'cosine', 'step'
        self.cls_weight_start = 0.5  # Starting classification weight
        self.cls_weight_end = 2.0    # Ending classification weight
        self.cls_weight_warmup_epochs = 50  # Epochs to reach target weight
        
        # Focal Loss weight adjustment
        self.use_focal_cls_weight = False
        self.focal_alpha = 2.0  # Focal loss alpha parameter
        self.focal_gamma = 2.0  # Focal loss gamma parameter  
        self.min_cls_weight = 0.1  # Minimum classification weight
        self.max_cls_weight = 5.0  # Maximum classification weight
        
        # Varifocal Loss configuration
        self.use_varifocal_loss = False
        self.varifocal_gamma = 2.0  # Varifocal loss gamma parameter
        self.varifocal_alpha = 0.75  # Varifocal loss alpha parameter
        
        # Standard Focal Loss configuration  
        self.use_focal_loss = False
        self.focal_gamma = 1.5  # Standard focal loss gamma parameter
        self.focal_alpha = 0.25  # Standard focal loss alpha parameter
        
        self.startEpoch = 0
        self.maxEpoch = 200
        self.backboneFreezeEpochs = []
        self.distilEpochs = []
        self.batchSize = 16
        self.optimizerType = "SGD"
        self.optimizerMomentum = 0.937
        self.optimizerWeightDecay = 5e-4
        # AdamW optimizer parameters
        self.optimizerBetas = (0.9, 0.999)  # AdamW beta parameters (beta1, beta2)
        self.optimizerEps = 1e-8  # AdamW epsilon parameter for numerical stability
        self.schedulerType = "COS"
        self.baseLearningRate = 1e-2
        self.minLearningRate = self.baseLearningRate * 1e-2
        self.epochValidation = True
        self.trainSelectedClasses = None
        self.distilSelectedClasses = None
        self.checkpointModelFile = None

        # eval setup
        self.testSelectedClasses = None
        self.minIou = 0.5
        self.paintImages = False

        # early stopping setup
        self.use_early_stopping = False
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001
        self.early_stopping_monitor = 'val_loss'  # 'val_loss' or 'train_loss'
        self.early_stopping_mode = 'min'  # 'min' for loss, 'max' for accuracy/mAP

        # dataset splits
        self.trainSplitName = "train"
        self.validationSplitName = "validation"
        self.testSplitName = "test"
        self.distilSplitName = "c10new"

        # enriched by factory
        self.mode = None
        self.root = None
        self.cfgname = None
        self.nobuf = False

    def enrichTags(self, tags):
        for tag in tags:
            tokens = tag.split("@")
            match tokens[0]:
                case "cuda":
                    self.device = torch.device("cuda:{}".format(tokens[1]))
                case "mps":
                    if torch.backends.mps.is_available():
                        self.device = torch.device("mps")
                    else:
                        print("Warning: MPS tag specified, but MPS is not available. Falling back to CPU.")
                        self.device = torch.device("cpu")
                case "batch":
                    self.batchSize = int(tokens[1])
                case "phase":
                    self.phase = tokens[1]  # Allow string phases like 'nano'
        return self

    def finalize(self, tags):
        self.enrichTags(tags)

        # Get username based on operating system
        if platform.system() == "Windows":
            self.user = os.getenv("USERNAME")
        else:
            self.user = os.getenv("USER")

        if self.user is None or len(self.user) == 0:
            raise ValueError("User not found")
        if self.root is None:
            raise ValueError("Root not set")
        if self.mode is None:
            raise ValueError("Mode not set")
        if self.cfgname is None:
            raise ValueError("Cfgname not set")
        if self.phase is None:
            raise ValueError("Phase not set")
        if self.inputShape[0] == 0 or self.inputShape[0] % 32 != 0 or self.inputShape[1] == 0 or self.inputShape[1] % 32 != 0:
            raise ValueError("Invalid input shape, must be positive mutiples of 32: inputShape={}".format(self.inputShape))

        self.cacheDir()
        self.downloadDir()
        self.evalDir()

        if self.imageDir is None:
            raise ValueError("Image directory not set")
        if self.annotationDir is None:
            raise ValueError("Annotation directory not set")
        if self.classList is None:
            raise ValueError("Class list not set")

        if isinstance(self.classList, str):
            with open(self.classList) as f:
                self.classList = [x.strip() for x in f.readlines()]
                self.classList = [x for x in self.classList if len(x) > 0]
        if len(self.classList) == 0:
            raise ValueError("Empty class list")
        self.nc = len(self.classList)

        subsetMap = {}
        for splitName, subset in self.subsetMap.items():
            if isinstance(subset, str):
                with open(subset) as f:
                    subset = [x.strip() for x in f.readlines()]
                    subset = [x for x in subset if len(x) > 0]
            subsetMap[splitName] = subset
        self.subsetMap = subsetMap

        return self

    def cacheDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "__cache__")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def downloadDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "__download__")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def evalDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "eval")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def modelSavePath(self):
        if self.epochValidation:
            return self.epochBestWeightsPath()
        else:
            return self.epochCachePath()

    def epochBestWeightsPath(self):
        return os.path.join(self.cacheDir(), "best_weights.pth")

    def epochCachePath(self):
        return os.path.join(self.cacheDir(), "last_epoch_weights.pth")

    def epochInfoPath(self):
        return os.path.join(self.cacheDir(), "info.txt")

    def plotDir(self):
        dirpath = os.path.join(self.root, self.user, self.cfgname, "plots")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath
