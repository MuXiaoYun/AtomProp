"""
Module for pretraining tasks for GNNs.
"""

from atomprop.utils.features import AtomFeaturize, AtomFeaturizeLoss 
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def nan_to_zero(name: str):
    """
    Decorator to makes sure output has not nan.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if torch.is_tensor(result) and torch.isnan(result).any():
                zero_tensor = torch.zeros_like(result)
                result = torch.where(torch.isnan(result), zero_tensor, result)
                print(f"{name} detected nan.")
            return result
        return wrapper
    return decorator

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

    @nan_to_zero("NodeAttrPrediction")
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

    def __init__(self, criterion=nn.CrossEntropyLoss()):
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

    @nan_to_zero("MaskedNodePrediction")
    def compute_loss(self):
        """
        Compute the loss for the task.
        """
        loss = self.criterion(self.preds, self.labels)
        return loss

    def get_metrics(self):
        """
        Compute metrics for the task.
        In this case, we uses loss.
        """
        return {
            'loss': self.criterion(self.preds, self.labels).item()
        }

class GraphMaskContrast:
    """
    Graph-level masked contrastive learning task.
    By making contrast between unmasked, less-masked, and more-masked graph representations.
    """

    def __init__(self, less_rate, more_rate, margin=1, p=2):
        self.anchor = None
        self.positive = None
        self.negative = None
        self.less_rate = less_rate
        self.more_rate = more_rate
        self.margin = margin
        self.p = p

    def set_embeddings(self, anchor, positive, negative):
        """
        Set embeddings for anchor, positive, and negative samples.
        """
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    @nan_to_zero("GraphMaskContrast")
    def compute_loss(self):
        """
        Compute the contrastive loss.
        """
        return nn.TripletMarginLoss(margin=self.margin, p=self.p)(self.anchor, self.positive, self.negative)**2

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
    By making contrast on graphs in the same batch, calculate InfoNCE loss.
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

    @nan_to_zero("BatchContrast")
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
        loss = F.cross_entropy(similarity_matrix, labels)
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

class ScaffoldContrast:
    """
    Scaffold-level contrastive learning using Supervised Contrastive Loss (SupCon).
    Each graph is an anchor; all graphs in the same scaffold group are positives.
    Implements the multi-positive extension of InfoNCE.
    """

    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.embeddings = None
        self.group_labels = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
        
    def set_group_label(self, group_labels: list[list[int]]):
        self.group_labels = group_labels

    @nan_to_zero("ScaffoldContrast")
    def compute_loss(self):
        if self.embeddings is None or self.group_labels is None:
            raise ValueError("Embeddings or group labels not set.")
        
        emb = self.embeddings
        device = emb.device
        batch_size = emb.size(0)
        
        if batch_size <= 1:
            return emb.sum() * 0.0

        labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        for scaffold_id, group in enumerate(self.group_labels):
            for idx in group:
                if 0 <= idx < batch_size:
                    labels[idx] = scaffold_id

        unique_labels = labels[labels != -1]
        if len(unique_labels) == 0 or len(torch.unique(unique_labels)) == len(unique_labels):
            # No positive pairs → return zero loss with gradient
            return emb.sum() * 0.0

        # Normalize embeddings
        emb_norm = F.normalize(emb, dim=1)
        # Similarity matrix: [batch_size, batch_size]
        sim_matrix = torch.matmul(emb_norm, emb_norm.t()) / self.temperature

        # Mask for positive pairs (including self)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (labels != -1).unsqueeze(0)  # [N, N]
        mask.fill_diagonal_(False)  # exclude self as positive (optional; you can keep it)

        # For numerical stability, subtract max per row
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute exp logits
        exp_logits = torch.exp(logits)

        # Denominator: sum over all negatives + positives (but not self if excluded)
        # Standard SupCon denominator includes ALL except self
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        denominator = torch.sum(exp_logits * neg_mask, dim=1)  # [N]

        # Numerator: sum over positives for each anchor
        numerator = torch.sum(exp_logits * mask, dim=1)  # [N]

        # Avoid log(0)
        valid = numerator > 0
        if not valid.any():
            return emb.sum() * 0.0

        # SupCon loss per anchor: -log( numerator / denominator )
        loss_per_anchor = -torch.log(numerator[valid] / (denominator[valid] + 1e-8))
        loss = loss_per_anchor.mean()

        return loss

    def get_metrics(self):
        return {}

class FunctionalGroupsPrediction:
    """
    Functional groups prediction task.
    This class computes losses and metrics for the task.
    """

    def __init__(self, criterion=nn.BCEWithLogitsLoss()):
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

    @nan_to_zero("FunctionalGroupsPrediction")
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