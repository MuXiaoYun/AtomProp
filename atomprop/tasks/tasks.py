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
        ra(label, pred) = 1 - (||label - pred||) / ||label||
        """
        return {
            'relative_accuracy': 1 - torch.mean(torch.norm(self.labels - self.preds) / (torch.norm(self.labels) + 1e-16)).item()
        }
