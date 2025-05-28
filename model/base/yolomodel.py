import os
import torch
import torch.nn as nn
from misc.log import log
from misc.bbox import makeAnchors
from model.base.backbone import Backbone
from model.base.neck import Neck
from model.base.head import DetectHead


class YoloModelPhaseSetup(object):
    @staticmethod
    def getModelWRN(phase):
        """
        Reference: resources/yolov8.jpg
        @return (w, r, n), note that n = 3 x d
        """
        if phase == "nano":
            return (0.25, 2, 1)
        if phase == "small":
            return (0.5, 2, 1)
        if phase == "medium":
            return (0.75, 1.5, 2)
        if phase == "large":
            return (1, 1, 3)
        if phase == "extended":
            return (1.25, 1, 3)
        raise ValueError("Invalid model phase: {}".format(phase))


class YoloModel(nn.Module):
    @classmethod
    def loadModelFromFile(cls, mcfg, modelFile):
        if modelFile is None:
            raise ValueError("Model file not set")
        if not os.path.exists(modelFile):
            raise ValueError("Model file not exists")
        model = cls(mcfg)
        model.load(modelFile)
        return model

    def __init__(self, mcfg):
        super(YoloModel, self).__init__()

        self.mcfg = mcfg
        self.inferenceMode = False

        # model layes
        w, r, n = YoloModelPhaseSetup.getModelWRN(mcfg.phase)
        self.backbone = Backbone(w, r, n)
        self.neck = Neck(w, r, n)
        self.head = DetectHead(w, r, self.mcfg.nc, self.mcfg.regMax)

        # model static data
        self.layerStrides = [8, 16, 32]
        self.outputShapes = (
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 8), int(self.mcfg.inputShape[1] / 8)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 16), int(self.mcfg.inputShape[1] / 16)),
            (self.mcfg.nc + self.mcfg.regMax * 4, int(self.mcfg.inputShape[0] / 32), int(self.mcfg.inputShape[1] / 32)),
        )
        self.anchorPoints, self.anchorStrides = makeAnchors([x[-2:] for x in self.outputShapes], self.layerStrides, 0.5)
        self.anchorPoints = self.anchorPoints.to(self.mcfg.device)
        self.anchorStrides = self.anchorStrides.to(self.mcfg.device)
        self.proj = torch.arange(self.mcfg.regMax, dtype=torch.float).to(self.mcfg.device)
        self.scaleTensor = torch.tensor(self.mcfg.inputShape, device=self.mcfg.device, dtype=torch.float)[[1, 0, 1, 0]]

    def getTrainLoss(self):
        from train.loss import DetectionLoss
        return DetectionLoss(self.mcfg, self)

    def setInferenceMode(self, flag):
        self.inferenceMode = flag
        if self.inferenceMode:
            self.eval()
        else:
            self.train()
        return self

    def freezeBackbone(self):
        log.inf("Freezing backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.inf("Backbone frozen.")

    def unfreezeBackbone(self):
        log.inf("Unfreezing backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True
        log.inf("Backbone unfrozen.")

    def forward(self, x, output_features=False):
        # If in inference mode AND not asked for features, run with no_grad and return only head outputs.
        if self.inferenceMode and not output_features:
            with torch.no_grad():
                # Call forwardInternal, which will return only head outputs if output_features is False
                return self.forwardInternal(x, output_features=False)
        # Otherwise (training, or inference but features are requested for teacher model), run normally.
        return self.forwardInternal(x, output_features=output_features)

    def forwardInternal(self, x, output_features=False):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            If output_features is False (default):
                xo: (B, nc + regMax * 4, 80, 80)
                yo: (B, nc + regMax * 4, 40, 40)
                zo: (B, nc + regMax * 4, 20, 20)
            If output_features is True:
                A tuple: ( (xo,yo,zo), (bb_feat0, bb_feat1, bb_feat2, bb_feat3), (neck_C, neck_X, neck_Y, neck_Z) )
                More precisely: ( (head_outs), (backbone_features), (neck_features) )
                Or flattened for DistillationDetectionLoss: (xo, yo, zo, bb_f0, bb_f1, bb_f2, bb_f3, neck_C, neck_X, neck_Y, neck_Z)
        """
        bb_feat0, bb_feat1, bb_feat2, bb_feat3 = self.backbone.forward(x)
        # neck_C is p4_fused, neck_X is p3_out, neck_Y is p4_out, neck_Z is p5_out
        neck_C, neck_X, neck_Y, neck_Z = self.neck.forward(bb_feat1, bb_feat2, bb_feat3)
        xo, yo, zo = self.head.forward(neck_X, neck_Y, neck_Z)

        if output_features:
            # This flattened tuple structure matches what YoloStudentModel provides
            # and what DistillationDetectionLoss expects for student/teacher outputs.
            return (xo, yo, zo, bb_feat0, bb_feat1, bb_feat2, bb_feat3, neck_C, neck_X, neck_Y, neck_Z)
        else:
            return xo, yo, zo

    def save(self, modelFile, verbose=False):
        torch.save(self.state_dict(), modelFile)
        if verbose:
            log.inf("Yolo model state saved at {}".format(modelFile))

    def loadBackboneWeights(self, url):
        pretrainedState = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu",
            model_dir=self.mcfg.downloadDir(),
            progress=False,
        )
        missingKeys, unexpectedKeys = self.backbone.load_state_dict(pretrainedState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model url, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, url))
        if len(missingKeys) > 0:
            log.red("Missing keys in model url:\nmissing={}\nurl={}".format(missingKeys, url))
            import pdb; pdb.set_trace()
        else:
            log.grey("Pretrained backbone weights loaded from url: {}".format(url))

    def load(self, modelFile):
        modelState = torch.load(modelFile, weights_only=True)
        missingKeys, unexpectedKeys = self.load_state_dict(modelState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            import pdb; pdb.set_trace()
        else:
            log.grey("Yolo model loaded from file: {}".format(modelFile))
