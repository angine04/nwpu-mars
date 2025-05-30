import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from misc.log import log
from tqdm import tqdm
from dl.vocdataset import VocDataset
from factory.modelfactory import MarsModelFactory
from train.opt import MarsOptimizerFactory
from train.sched import MarsLearningRateSchedulerFactory
from .ema import ModelEMA 


class MarsBaseTrainer(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.cacheDir = os.path.join(self.mcfg.getCacheDir())
        self.epochInfoFile = os.path.join(self.cacheDir, "last_epoch.info")
        self.epochCacheFile = os.path.join(self.cacheDir, "last_weights.pth")
        self.bestCacheFile = os.path.join(self.cacheDir, "best_weights.pth")
        self.logFile = os.path.join(self.cacheDir, "train.log")
        self.lossFile = os.path.join(self.cacheDir, "losses.log")
        
        # loss tracking
        self.bestLoss = np.nan
        
        # backbone frozen logic
        self.backboneFrozen = False
        self.train_losses = []
        self.val_losses = []
        self.ema = None  # 新增EMA引用

    def initTrainDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.trainSplitName, isTest=False, fullInfo=False, selectedClasses=self.mcfg.trainSelectedClasses)

    def initValidationDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.validationSplitName, isTest=False, fullInfo=False, selectedClasses=None)

    def initTestDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.testSplitName, isTest=False, fullInfo=False, selectedClasses=None)

    def initModel(self):
        import platform
        import torch
        from util import log

        # Check if last epoch info file exists
        if os.path.exists(self.epochInfoFile):
            model, startEpoch = MarsModelFactory.loadCheckpointModel(self.mcfg, self.epochCacheFile)
            try:
                with open(self.epochInfoFile, 'r') as file:
                    content = file.read().strip()
                startEpoch = int(content)
            except Exception as e:
                log.red("Failed to load last epoch info from file: {}".format(self.epochInfoFile))
                raise ValueError("Failed to load last epoch info from file: {}".format(self.epochInfoFile))
            if startEpoch < self.mcfg.maxEpoch:
                log.yellow("Checkpoint loaded: resuming from epoch {}".format(startEpoch))
            
            # 🔧 修复：EMA初始化移到模型完全加载后
            if getattr(self.mcfg, 'use_ema', False):
                ema_decay = getattr(self.mcfg, 'ema_decay', 0.9999)
                ema_tau = getattr(self.mcfg, 'ema_tau', 2000)
                self.ema = ModelEMA(model, decay=ema_decay, tau=ema_tau)
                log.green(f"EMA initialized with decay={ema_decay}, tau={ema_tau}")
            
            return model, startEpoch

        if self.mcfg.checkpointModelFile is not None: # use model from previous run, but start epoch from zero
            model, _ = MarsModelFactory.loadCheckpointModel(self.mcfg, self.mcfg.checkpointModelFile)
        else:
            model = MarsModelFactory.loadNewModel(self.mcfg)
        
        # 🔧 修复：EMA初始化移到模型完全准备好之后
        if getattr(self.mcfg, 'use_ema', False):
            ema_decay = getattr(self.mcfg, 'ema_decay', 0.9999)
            ema_tau = getattr(self.mcfg, 'ema_tau', 2000)
            self.ema = ModelEMA(model, decay=ema_decay, tau=ema_tau)
            log.green(f"EMA initialized with decay={ema_decay}, tau={ema_tau}")
        
        return model, 0

    def initLoss(self, model):
        return model.getTrainLoss()

    def initOptimizer(self, model):
        return MarsOptimizerFactory.initOptimizer(self.mcfg, model)

    def initScheduler(self, opt):
        return MarsLearningRateSchedulerFactory.initScheduler(self.mcfg, opt)

    def preEpochSetup(self, model, epoch):
        if self.mcfg.backboneFreezeEpochs is not None:
            if epoch in self.mcfg.backboneFreezeEpochs:
                model.freezeBackbone()
                self.backboneFrozen = True
            else:
                model.unfreezeBackbone()
                self.backboneFrozen = False

    def fitOneEpoch(self, model, loss, dataLoader, optimizer, epoch):
        trainLoss = 0
        model.setInferenceMode(False)
        numBatches = int(len(dataLoader.dataset) / dataLoader.batch_size)
        progressBar = tqdm(total=numBatches, desc="Epoch {}/{}".format(epoch + 1, self.mcfg.maxEpoch), postfix=dict, mininterval=0.5, ascii=False, ncols=130)

        for batchIndex, batch in enumerate(dataLoader):
            images, labels = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)
            optimizer.zero_grad()

            output = model(images)
            stepLoss = loss(output, labels)
            stepLoss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if self.ema is not None:
                self.ema.update(model)  # 每个batch后更新EMA

            trainLoss += stepLoss.item()
            progressBar.set_postfix(trainLossPerBatch=trainLoss / (batchIndex + 1), backboneFrozen=self.backboneFrozen)
            progressBar.update(1)

        progressBar.close()
        avg_train_loss = trainLoss / numBatches if numBatches > 0 else 0
        self.train_losses.append(avg_train_loss)
        return trainLoss

    def epochValidation(self, model, loss, dataLoader, epoch):
        eval_model = self.ema.ema if (self.ema is not None) else model
        eval_model.setInferenceMode(True)
        
        if not self.mcfg.epochValidation:
            return np.nan

        validationLoss = 0
        model.setInferenceMode(True)
        numBatches = int(len(dataLoader.dataset) / dataLoader.batch_size)
        progressBar = tqdm(total=numBatches, desc="Epoch {}/{}".format(epoch + 1, self.mcfg.maxEpoch), postfix=dict, mininterval=0.5, ascii=False, ncols=100)

        for batchIndex, batch in enumerate(dataLoader):
            images, labels = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)

            output = eval_model(images)
            stepLoss = loss(output, labels)

            validationLoss += stepLoss.item()
            progressBar.set_postfix(validationLossPerBatch=validationLoss / (batchIndex + 1))
            progressBar.update(1)

        progressBar.close()
        avg_val_loss = validationLoss / numBatches if numBatches > 0 else 0
        self.val_losses.append(avg_val_loss)
        return validationLoss

    def run(self):
        log.cyan("Mars trainer running...")

        model, startEpoch = self.initModel()
        if startEpoch >= self.mcfg.maxEpoch:
            log.inf("Training skipped")
            return

        loss = self.initLoss(model)
        opt = self.initOptimizer(model)
        scheduler = self.initScheduler(opt)
        trainLoader = self.initTrainDataLoader()
        validationLoader = self.initValidationDataLoader()

        for epoch in range(startEpoch, self.mcfg.maxEpoch):
            self.preEpochSetup(model, epoch)
            scheduler.updateLearningRate(epoch)
            trainLoss = self.fitOneEpoch(
                model=model,
                loss=loss,
                dataLoader=trainLoader,
                optimizer=opt,
                epoch=epoch,
            )
            validationLoss = self.epochValidation(
                model=model,
                loss=loss,
                dataLoader=validationLoader,
                epoch=epoch,
            )
            self.epochSave(epoch, model, trainLoss, validationLoss)

        log.inf("Mars trainer finished with max epoch at {}".format(self.mcfg.maxEpoch))
        if self.train_losses:
            self.plot_loss_curves()

    def epochSave(self, epoch, model, trainLoss, validationLoss):
        save_model = self.ema.ema if (self.ema is not None) else model
        save_model.save(self.epochCacheFile)
        
        if self.mcfg.epochValidation and (np.isnan(self.bestLoss) or validationLoss < self.bestLoss):
            log.green("Caching best weights at epoch {}...".format(epoch + 1))
            self.bestLoss = validationLoss
            save_model.save(self.bestCacheFile)  # 保存最佳EMA模型

        with open(self.epochInfoFile, 'w') as file:
            file.write(str(epoch + 1))

    def plot_loss_curves(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-o', label='Training Loss')
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-s', label='Validation Loss')
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.mcfg.plotDir(), "loss_curve.png")
        plt.savefig(plot_path)
        log.inf(f"Loss curve saved to {plot_path}")
        plt.close()


def getTrainer(mcfg):
    return MarsBaseTrainer(mcfg)
