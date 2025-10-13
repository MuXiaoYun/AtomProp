"""
Module for pretraining tasks for GNNs.
"""

from atomprop.utils.features import atomFeaturize  

class NodeAttrPrediction:
    """
    Node attribute prediction task.
    This class computes losses and metrics for the task.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        self.preds = None
        self.labels = None

    def run_pred(self, data, pmodel, fmodel):
        """
        Process a batch of data and generate predictions.
        pmodel: pre-trained model (frozen)
        fmodel: downstream model (trainable)
        """
        pmodel.eval()
        fmodel.train()
        embs = pmodel(data)
        preds = fmodel(embs)
        self.preds = preds
        return

    def run_label(self, data):
        """
        Process a batch of data and generate labels.
        """
        self.labels = torch.tensor([atomFeaturize.featurize(mol) for mol in data])
        return

    def compute_loss(self):
        """
        Compute the loss for the task.
        """
        loss = self.criterion(preds_tensor, labels_tensor)
        return loss

    def get_metrics(self):
        """
        Compute metrics for the task.
        In this case, we compute relative accuracy.
        """
        return {
            'relative_accuracy': (self.preds == self.labels).sum().item() / len(self.labels)
        }