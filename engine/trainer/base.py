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
        self.epochInfoFile = self.mcfg.epochInfoPath()
        self.epochCacheFile = self.mcfg.epochCachePath()
        self.bestCacheFile = self.mcfg.epochBestWeightsPath()
        self.logFile = os.path.join(self.mcfg.cacheDir(), "train.log")
        self.lossFile = os.path.join(self.mcfg.cacheDir(), "losses.log")
        
        # loss tracking
        self.bestLoss = np.nan
        self.bestLossEpoch = np.nan
        
        # checkpoint files for resuming training
        self.checkpointFiles = [
            self.epochCacheFile,
            self.epochInfoFile,
        ]
        if self.mcfg.epochValidation:
            self.checkpointFiles.append(self.bestCacheFile)
        
        # backbone frozen logic
        self.backboneFrozen = False
        self.train_losses = []
        self.val_losses = []
        self.ema = None  # EMA reference

    def initTrainDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.trainSplitName, isTest=False, fullInfo=False, selectedClasses=self.mcfg.trainSelectedClasses)

    def initValidationDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.validationSplitName, isTest=False, fullInfo=False, selectedClasses=None)

    def initTestDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.testSplitName, isTest=False, fullInfo=False, selectedClasses=None)

    def initModel(self):
        startEpoch = 0 # Default start epoch
        model = None # Initialize model to None
        loaded_from_checkpoint = False

        # Check if all necessary checkpoint files exist
        if not self.mcfg.nobuf and all(os.path.exists(x) for x in self.checkpointFiles):
            log.yellow("Attempting to resume training from checkpoint...")
            try:
                # loadPretrainedModel returns model only, not epoch
                model = MarsModelFactory.loadPretrainedModel(self.mcfg, self.epochCacheFile)

                with open(self.epochInfoFile, 'r') as file:
                    content = file.read().strip()
                # Read detailed training information
                epoch_found_in_file = False
                for line in content.splitlines():
                    tokens = line.split('=')
                    if len(tokens) == 2:
                        key, value = tokens
                        if key == 'last_saved_epoch':
                            startEpoch = int(value)
                            epoch_found_in_file = True
                        elif key == 'best_loss':
                            try:
                                self.bestLoss = float(value)
                            except ValueError:
                                self.bestLoss = np.nan # Handle NaN string
                        elif key == 'best_loss_epoch':
                            try:
                                self.bestLossEpoch = int(value)
                            except ValueError:
                                self.bestLossEpoch = np.nan # Handle NaN string

                if epoch_found_in_file and startEpoch < self.mcfg.maxEpoch:
                    log.yellow("Checkpoint loaded: resuming from epoch {}".format(startEpoch))
                    loaded_from_checkpoint = True
                else:
                     log.yellow("Checkpoint found but epoch info incomplete or max epoch reached, starting from epoch 0.")


            except Exception as e:
                log.red("Failed to load checkpoint from {}: {}".format(self.epochCacheFile, e))
                # If checkpoint loading fails, fall through to load new model or external checkpoint
                model = None # Ensure model is None to trigger loading below
                startEpoch = 0
                loaded_from_checkpoint = False

        # If not loaded from checkpoint, check for external checkpoint or load new model
        if not loaded_from_checkpoint:
            if self.mcfg.checkpointModelFile is not None:
                 # Load model from specified external checkpoint file (start epoch 0)
                 log.yellow("Loading model from external checkpoint: {}".format(self.mcfg.checkpointModelFile))
                 model = MarsModelFactory.loadPretrainedModel(self.mcfg, self.mcfg.checkpointModelFile)
                 startEpoch = 0
            else:
                # Load a brand new model (potentially with pretrained backbone)
                log.yellow("Loading a new model...")
                model = MarsModelFactory.loadNewModel(self.mcfg, self.mcfg.pretrainedBackboneUrl)
                startEpoch = 0

        # Ensure model is loaded before EMA initialization
        if model is None:
             raise RuntimeError("Model failed to initialize.")

        # Initialize EMA if enabled
        if getattr(self.mcfg, 'use_ema', False):
            ema_decay = getattr(self.mcfg, 'ema_decay', 0.9999)
            ema_tau = getattr(self.mcfg, 'ema_tau', 2000)
            self.ema = ModelEMA(model, decay=ema_decay, tau=ema_tau)
            log.green(f"EMA initialized with decay={ema_decay}, tau={ema_tau}")

        return model, startEpoch

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
                self.ema.update(model)  # Update EMA after each batch

            trainLoss += stepLoss.item()
            progressBar.set_postfix(trainLossPerBatch=trainLoss / (batchIndex + 1), backboneFrozen=self.backboneFrozen)
            progressBar.update(1)

        progressBar.close()
        avg_train_loss = trainLoss / numBatches if numBatches > 0 else 0
        self.train_losses.append(avg_train_loss)
        return trainLoss

    def epochValidation(self, model, loss, dataLoader, epoch):
        if not self.mcfg.epochValidation:
            return np.nan

        eval_model = self.ema.ema if (self.ema is not None) else model
        eval_model.setInferenceMode(True)
        
        validationLoss = 0
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
        # Set up log file for training
        log.setLogFile(self.logFile)
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
        # Save detailed training information to epochInfoFile
        with open(self.epochInfoFile, "w") as f:
            f.write("last_saved_epoch={}\n".format(epoch + 1))
            f.write("train_loss={}\n".format(trainLoss))
            f.write("validation_loss={}\n".format(validationLoss if not np.isnan(validationLoss) else 'NaN'))
            f.write("best_loss_epoch={}\n".format(self.bestLossEpoch if not np.isnan(self.bestLossEpoch) else 'NaN'))
            f.write("best_loss={}\n".format(self.bestLoss if not np.isnan(self.bestLoss) else 'NaN'))

        # Log losses to losses.log file
        with open(self.lossFile, "a") as f:
            f.write("epoch={}, train_loss={:.6f}, val_loss={:.6f}\n".format(
                epoch + 1, 
                trainLoss, 
                validationLoss if not np.isnan(validationLoss) else float('nan')
            ))

        # Determine model to save (EMA preferred)
        save_model = self.ema.ema if (self.ema is not None) else model

        # Check if this is the best validation loss
        is_best = self.mcfg.epochValidation and (np.isnan(self.bestLoss) or validationLoss < self.bestLoss)
        
        if is_best:
            log.green("Caching best weights at epoch {}...".format(epoch + 1))
            self.bestLoss = validationLoss
            self.bestLossEpoch = epoch + 1
            save_model.save(self.bestCacheFile)  # Save best model
        
        # Always save current epoch weights to epochCacheFile (for resuming training)
        save_model.save(self.epochCacheFile)

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
