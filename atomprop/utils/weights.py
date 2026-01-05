"""
Module for weight stratergies among pretraining tasks.
"""
import torch
from atomprop.utils.grads_utils import find_optimal_weights
import torch.nn as nn

log_var_lower_bound = -10
log_var_upper_bound = 10

class GradNorm(nn.Module):
    def __init__(self, num_tasks, init_weights=None, alpha=0.16):
        """
        Args:
            num_tasks (int): number of tasks T
            init_weights (list or Tensor, optional): init weights or from Uncertainty Weighting
            alpha (float): loss ratio sensitivity 
        """
        super().__init__()
        if init_weights is None:
            init_weights = torch.ones(num_tasks)
        else:
            init_weights = init_weights
        self.task_weights = nn.Parameter(init_weights)
        self.alpha = alpha
        self.num_tasks = num_tasks

    def forward(self, task_losses, shared_params):
        """
        Calculate GradNorm loss. 
        Args:
            task_losses (List[Tensor]): losses for each task, length T
            shared_params (Iterable[nn.Parameter]): parameters of shared layers
        Returns:
            total_loss (Tensor): total loss
            gradnorm_loss (Tensor): GradNorm loss
            normalized_weights (Tensor): normed weights
        """
        T = self.num_tasks
        weights = self.task_weights
        normalized_weights = weights / weights.mean()  # mean=1 => sum=T

        weighted_losses = [normalized_weights[i] * task_losses[i] for i in range(T)]
        total_loss = sum(weighted_losses)

        G = []
        for i in range(T):
            grad = torch.autograd.grad(
                outputs=normalized_weights[i] * task_losses[i],
                inputs=shared_params,
                retain_graph=True,  
                allow_unused=True, 
                create_graph=False   
            )
            grad_flat = torch.cat([g.flatten() for g in grad if g is not None])
            G_i = torch.norm(grad_flat, p=2)
            G.append(G_i)
        G = torch.stack(G)  # shape: [T]

        G_avg = G.mean()
        with torch.no_grad():
            L = torch.stack(task_losses)
            L_avg = L.mean()
            r_target = (L / L_avg) ** self.alpha

        gradnorm_loss = torch.abs(G - G_avg * r_target).sum() 

        return total_loss, gradnorm_loss, normalized_weights.detach()
    
class ParetoOpt:
    """
    Given a group of grads g1, g2, ..., gn, this method calculates w1, w2, ..., wn so that g_opt = Σwi*gi satisfies g_opt = argmin max ∠(g_opt, gi).
    """
    def __init__(self, task_num, device):
        self.task_num = task_num
        self.device = device
    
    def outputs(self, grads: list):
        grads_mean = []
        for grad in grads:
            grads_mean.append(torch.mean(grad, dim=0))
        ws = find_optimal_weights(grads_mean)
        return ws.to(self.device)

class UncertaintyWeighting(nn.Module):
    """
    Uncertainty weight stratergy.
    Reference: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        """
        losses: list of task losses [L1, L2, ..., Ln]
        loss_types: list of loss types [T1, T2, ..., Tn], type in ["classification", "regression"]
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            log_var_clamped = torch.clamp(self.log_vars[i], log_var_lower_bound, log_var_upper_bound)
            precision = torch.exp(-log_var_clamped)
            total_loss += 0.5 * precision * loss + 0.5 * log_var_clamped
        return total_loss    
    
class FixedUncertaintyWeighting(nn.Module):
    """
    Fixed uncertainty weight stratergy.
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses, log_vars):
        """
        losses: list of task losses [L1, L2, ..., Ln]
        log_vars: list of loss log vars
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-log_vars[i])
            total_loss += precision * loss
        return total_loss    