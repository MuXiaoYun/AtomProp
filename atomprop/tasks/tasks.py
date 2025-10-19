"""
Module for pretraining tasks for GNNs.
"""

from atomprop.utils.features import atomFeaturize  
import torch

class NodeAttrPrediction:
    """
    Node attribute prediction task.
    This class computes losses and metrics for the task.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def set_pred(self, preds):
        """
        Set predictions.
        """
        self.preds = preds
        return

    def run_label(self, data, device):
        """
        Process a batch of data and generate labels.
        """
        self.labels = torch.cat([atomFeaturize.featurize(mol) for mol in data], dim=0).to(device)
        return

    def compute_loss(self):
        """
        Compute the loss for the task.
        """
        loss = self.criterion(self.preds, self.labels)
        return loss

    def get_metrics(self):
        """
        Compute metrics for the task.
        In this case, we compute relative accuracy.
        """
        return {
            'relative_accuracy': (self.preds == self.labels).sum().item() / len(self.labels)
        }
