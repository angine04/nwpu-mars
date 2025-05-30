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
from .early_stopping import EarlyStopping
from train.loss_weight_scheduler import LossWeightScheduler, FocalLossWeightScheduler


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
        self.early_stopping = None  # Early stopping reference
        
        # Initialize loss weight scheduler (choose between regular and focal)
        if getattr(self.mcfg, 'use_focal_cls_weight', False):
            self.loss_weight_scheduler = FocalLossWeightScheduler(self.mcfg)
        else:
            self.loss_weight_scheduler = LossWeightScheduler(self.mcfg)
        
        # Initialize early stopping if enabled
        if getattr(self.mcfg, 'use_early_stopping', False):
            self.early_stopping = EarlyStopping(
                patience=getattr(self.mcfg, 'early_stopping_patience', 10),
                min_delta=getattr(self.mcfg, 'early_stopping_min_delta', 0.001),
                monitor=getattr(self.mcfg, 'early_stopping_monitor', 'val_loss'),
                mode=getattr(self.mcfg, 'early_stopping_mode', 'min'),
                restore_best_weights=True
            )
            log.green(f"Early stopping initialized: patience={self.early_stopping.patience}, "
                     f"monitor={self.early_stopping.monitor}, mode={self.early_stopping.mode}")

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
                        # Restore early stopping state
                        elif key == 'early_stopping_wait' and self.early_stopping is not None:
                            try:
                                self.early_stopping.wait = int(value)
                            except ValueError:
                                pass
                        elif key == 'early_stopping_best_epoch' and self.early_stopping is not None:
                            try:
                                self.early_stopping.best_epoch = int(value) - 1  # Convert back to 0-indexed
                            except ValueError:
                                pass
                        elif key == 'early_stopping_best_value' and self.early_stopping is not None:
                            try:
                                self.early_stopping.best = float(value)
                            except ValueError:
                                pass

                if epoch_found_in_file and startEpoch < self.mcfg.maxEpoch:
                    log.yellow("Checkpoint loaded: resuming from epoch {}".format(startEpoch))
                    if self.early_stopping is not None:
                        log.yellow(f"Early stopping state restored: wait={self.early_stopping.wait}, "
                                  f"best_epoch={self.early_stopping.best_epoch + 1}, "
                                  f"best_value={self.early_stopping.best:.6f}")
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

        # 用于Focal Loss调度器的损失累积
        epoch_cls_loss = 0
        epoch_total_loss = 0
        focal_update_interval = max(1, numBatches // 10)  # 每10%的batch更新一次权重

        for batchIndex, batch in enumerate(dataLoader):
            images, labels = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)
            optimizer.zero_grad()

            output = model(images)
            
            # 如果使用Focal Loss调度器，获取损失组件
            if hasattr(self.loss_weight_scheduler, 'use_focal_weight') and self.loss_weight_scheduler.use_focal_weight:
                stepLoss, box_loss, cls_loss, dfl_loss = loss.get_loss_components(output, labels)
                epoch_cls_loss += cls_loss
                epoch_total_loss += (box_loss + cls_loss + dfl_loss)
                
                # 定期更新Focal权重
                if (batchIndex + 1) % focal_update_interval == 0:
                    avg_cls_loss = epoch_cls_loss / (batchIndex + 1)
                    avg_total_loss = epoch_total_loss / (batchIndex + 1)
                    self.loss_weight_scheduler.update_loss_weights(epoch, avg_cls_loss, avg_total_loss)
            else:
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

    def _version_log_file(self, log_file_path):
        """Versions a log file by appending a number if it already exists."""
        if not os.path.exists(log_file_path):
            return

        base, ext = os.path.splitext(log_file_path)
        i = 1
        # Find the next available version number
        while os.path.exists(f"{base}.{i}{ext}"):
            i += 1

        new_log_file_path = f"{base}.{i}{ext}"
        os.rename(log_file_path, new_log_file_path)
        log.yellow(f"Renamed existing log file {log_file_path} to {new_log_file_path}")

    def run(self):
        # Version existing log files before starting new run
        self._version_log_file(self.logFile)
        self._version_log_file(self.lossFile)

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
            # Update loss weights for current epoch
            self.loss_weight_scheduler.update_loss_weights(epoch)
            self.loss_weight_scheduler.log_current_weights(epoch)
            
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
            
            # Early stopping check
            if self.early_stopping is not None:
                # Determine which metric to monitor
                if self.early_stopping.monitor == 'val_loss':
                    monitor_value = validationLoss
                elif self.early_stopping.monitor == 'train_loss':
                    monitor_value = trainLoss
                else:
                    log.red(f"Unknown monitor metric: {self.early_stopping.monitor}")
                    monitor_value = validationLoss
                
                # Check if training should stop
                if self.early_stopping(monitor_value, epoch):
                    log.inf(f"Early stopping triggered at epoch {epoch + 1}")
                    break

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
            
            # Save early stopping information
            if self.early_stopping is not None:
                f.write("early_stopping_wait={}\n".format(self.early_stopping.wait))
                f.write("early_stopping_best_epoch={}\n".format(self.early_stopping.best_epoch + 1))
                f.write("early_stopping_best_value={}\n".format(self.early_stopping.best))
                f.write("early_stopping_monitor={}\n".format(self.early_stopping.monitor))

        # Log losses to losses.log file
        with open(self.lossFile, "a") as f:
            early_stop_info = ""
            if self.early_stopping is not None:
                early_stop_info = f", early_stop_wait={self.early_stopping.wait}/{self.early_stopping.patience}"
            
            f.write("epoch={}, train_loss={:.6f}, val_loss={:.6f}{}\n".format(
                epoch + 1, 
                trainLoss, 
                validationLoss if not np.isnan(validationLoss) else float('nan'),
                early_stop_info
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
        plt.yscale('log')
        plot_path = os.path.join(self.mcfg.plotDir(), "loss_curve.png")
        plt.savefig(plot_path)
        log.inf(f"Loss curve saved to {plot_path}")
        plt.close()


def getTrainer(mcfg):
    return MarsBaseTrainer(mcfg)
