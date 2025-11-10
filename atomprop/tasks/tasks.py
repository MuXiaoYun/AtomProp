"""
Module for pretraining tasks for GNNs.
"""

from atomprop.utils.features import AtomFeaturize, AtomFeaturizeLoss 
import torch
import torch.nn as nn

class NodeAttrPrediction:
    """
    Node attribute prediction task.
    This class computes losses and metrics for the task.
    """

    def __init__(self, criterion=AtomFeaturizeLoss()):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def set_pred(self, preds):
        """
        Set predictions.
        """
        self.preds = preds
        return

    def run_label(self, mol_batch, device):
        """
        Process a batch of data and generate labels.
        """
        self.labels = torch.cat([AtomFeaturize.featurize(mol) for mol in mol_batch], dim=0).to(device)
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

class MaskedNodePrediction:
    """
    Masked node type prediction task.
    """

    def __init__(self, criterion=nn.MSELoss()):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def set_pred(self, preds):
        """
        Set predictions.
        """
        self.preds = preds
        return

    def set_label(self, labels):
        """
        Set labels.
        """
        self.labels = labels
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

class GraphMaskContrast:
    """
    Graph-level masked contrastive learning task.
    By making contrast between unmasked, less-masked, and more-masked graph representations.
    """

    def __init__(self, less_rate, more_rate):
        self.anchor = None
        self.positive = None
        self.negative = None
        self.less_rate = less_rate
        self.more_rate = more_rate

    def set_embeddings(self, anchor, positive, negative):
        """
        Set embeddings for anchor, positive, and negative samples.
        """
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def compute_loss(self):
        """
        Compute the contrastive loss.
        """
        return nn.TripletMarginLoss(margin=1, p=2)(self.anchor, self.positive, self.negative)

    def get_metrics(self):
        """
        Compute metrics for the task.
        In this case, we compute relative accuracy.
        ra(anchor, positive, negative) = 1 + less_rate/more_rate - (||anchor - positive||) / ||anchor - negative||
        : param less_rate: The masking rate for the positive sample.
        : param more_rate: The masking rate for the negative sample.
        """
        return {
            'relative_accuracy': 1 + self.less_rate/self.more_rate - torch.mean(torch.norm(self.anchor - self.positive) / (torch.norm(self.anchor - self.negative) + 1e-16)).item()
        }

class BatchContrast:
    """
    Graph-level masked contrastive learning task.
    By making contrast on graphs in the same batch.
    """

    def __init__(self, temperature=0.1):
        self.anchor = None
        self.negative = None
        self.temperature = temperature

    def set_embeddings(self, anchor, negative):
        """
        Set embeddings for anchor and negative samples.
        """
        self.anchor = anchor
        self.negative = negative

    def compute_loss(self):
        """
        Compute the contrastive loss.
        Using InfoNCE loss.
        We expect anchor and negative to be shape of (batch_size, embedding_dim)
        """
        batch_size = self.anchor.size(0)
        anchor_norm = self.anchor / self.anchor.norm(dim=1, keepdim=True)
        negative_norm = self.negative / self.negative.norm(dim=1, keepdim=True)
        similarity_matrix = torch.matmul(anchor_norm, negative_norm.t()) / self.temperature  # (batch_size, batch_size)
        labels = torch.arange(batch_size).to(self.anchor.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

    def get_metrics(self):
        """
        Compute metrics for the task.
        In this case, we compute relative accuracy.
        ra(anchor, negative) = 1 - (1/batch_size) * sum(||anchor_i - negative_i||) / sum(||anchor_i - negative_j||)
        """
        batch_size = self.anchor.size(0)
        total_positive_distance = torch.sum(torch.norm(self.anchor - self.negative, dim=1))
        total_negative_distance = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    total_negative_distance += torch.norm(self.anchor[i] - self.negative[j])
        relative_accuracy = 1 - (total_positive_distance / (total_negative_distance + 1e-16))
        return {
            'relative_accuracy': relative_accuracy.item()
        }

class BondLengthPrediction:
    pass

class BondAnglePrediction:
    pass

class DFTPrediction:
    """
    Graph level prediction of DFT properties.
    """

    def __init__(self, criterion=nn.MSELoss()):
        self.criterion = criterion
        self.preds = None
        self.labels = None
