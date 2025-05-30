import numpy as np
from misc.log import log


class EarlyStopping:
    """Early stopping utility to stop training when monitored metric stops improving.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        monitor (str): Quantity to be monitored. Options: 'val_loss', 'train_loss'.
        mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity 
                   monitored has stopped decreasing; in 'max' mode it will stop when the quantity
                   monitored has stopped increasing.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
    """
    
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'.")
    
    def __call__(self, current_value, epoch):
        """Check if training should be stopped.
        
        Args:
            current_value (float): Current value of the monitored metric.
            epoch (int): Current epoch number.
            
        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if np.isnan(current_value):
            log.yellow(f"Early stopping: {self.monitor} is NaN at epoch {epoch + 1}")
            return False
            
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.best_epoch = epoch
            self.wait = 0
            log.green(f"Early stopping: {self.monitor} improved to {current_value:.6f} at epoch {epoch + 1}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                log.red(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs. "
                       f"Best {self.monitor}: {self.best:.6f} at epoch {self.best_epoch + 1}. "
                       f"Stopping training at epoch {epoch + 1}.")
                return True
            else:
                log.yellow(f"Early stopping: {self.monitor} did not improve ({current_value:.6f}). "
                          f"Patience: {self.wait}/{self.patience}")
        
        return False
    
    def get_best_epoch(self):
        """Get the epoch number with the best monitored value."""
        return self.best_epoch
    
    def get_best_value(self):
        """Get the best monitored value."""
        return self.best
    
    def reset(self):
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf 