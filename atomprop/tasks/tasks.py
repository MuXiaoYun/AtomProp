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

class BondAnglePrediction:
    """
    Sub-structure level bond angle prediction task.
    3 atoms form a bond angle.
    1st atom -- center atom -- 3rd atom
    """

    def __init__(self, criterion=nn.MSELoss()):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def set_pred(self, preds):
        """
        Set preds.
        """
        self.preds = preds
        return
        
    def set_label(self, xyzs, indices):
        """
        Calculate angles and set labels.
        Args:
            xyzs: [N, 3] 
            indices: [S, 3] 
        Returns:
            coss: [S, 1]
        """
        triplets = xyzs[indices]  # [S, 3, 3]
        
        p0 = triplets[:, 0]
        p1 = triplets[:, 1]
        p2 = triplets[:, 2]
        
        v1 = p1 - p0  
        v2 = p2 - p0  
        
        dot_products = torch.sum(v1 * v2, dim=1, keepdim=True)  # [S, 1]

        norm_v1 = torch.norm(v1, dim=1, keepdim=True)  # [S, 1]
        norm_v2 = torch.norm(v2, dim=1, keepdim=True)  # [S, 1]

        eps = 1e-8
        cosines = dot_products / (norm_v1 * norm_v2 + eps)  # [S, 1]

        cosines = torch.clamp(cosines, -1.0, 1.0)
        self.labels = cosines
        return

    def compute_loss(self):
        """
        Compute the loss for the task.
        To note, thre are NANs in self.labels, which are angles that cannot be defined. We need to mask them and those relevant in self.preds.
        """
        mask = ~torch.isnan(self.labels)
        filtered_preds = self.preds[mask]
        filtered_labels = self.labels[mask]
        loss = self.criterion(filtered_preds, filtered_labels)
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

class DihedralAnglePrediction:
    """
    Sub-structure level dihedral angle prediction task.
    4 atoms form a dihedral angle.
    1st atom -- center atom1 -- center atom2 -- 4th atom
    """
    def __init__(self, criterion=nn.MSELoss()):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def set_pred(self, preds):
        """
        Set preds.
        """
        self.preds = preds
        return

    def set_label(self, xyzs, indices):
        """
        Calculate dihedral angles and set labels.
        Args:
            xyzs: [N, 3] 
            indices: [S, 4] 
        Returns:
            coss: [S, 1]
        """
        quadruplets = xyzs[indices]  # [S, 4, 3]
        
        p0 = quadruplets[:, 0]
        p1 = quadruplets[:, 1]
        p2 = quadruplets[:, 2]
        p3 = quadruplets[:, 3]
        
        b0 = p1 - p0  
        b1 = p2 - p1  
        b2 = p3 - p2  
        
        b1_norm = b1 / (torch.norm(b1, dim=1, keepdim=True) + 1e-8)
        
        v = b0 - torch.sum(b0 * b1_norm, dim=1, keepdim=True) * b1_norm
        w = b2 - torch.sum(b2 * b1_norm, dim=1, keepdim=True) * b1_norm
        
        x = torch.sum(v * w, dim=1, keepdim=True)  # [S, 1]
        y = torch.sum(torch.cross(b1_norm, v, dim=1) * w, dim=1, keepdim=True)  # [S, 1]

        angles = torch.atan2(y, x)  # [S, 1]
        cosines = torch.cos(angles)  # [S, 1]

        self.labels = cosines
        return

    def compute_loss(self):
        """
        Compute the loss for the task.
        To note, thre are NANs in self.labels, which are angles that cannot be defined. We need to mask them and those relevant in self.preds.
        """
        mask = ~torch.isnan(self.labels)
        filtered_preds = self.preds[mask]
        filtered_labels = self.labels[mask]
        loss = self.criterion(filtered_preds, filtered_labels)
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
