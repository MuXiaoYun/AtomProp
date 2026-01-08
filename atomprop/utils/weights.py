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
            init_weights (list or Tensor, optional): initial task weights
            alpha (float): sensitivity to relative loss magnitudes
        """
        super().__init__()
        if init_weights is None:
            init_weights = torch.ones(num_tasks)
        else:
            init_weights = torch.as_tensor(init_weights, dtype=torch.float32)
        self.task_weights = nn.Parameter(init_weights)
        self.alpha = alpha
        self.num_tasks = num_tasks

    def forward(self, task_losses, shared_params):
        """
        Compute total loss and GradNorm regularization loss.
        This version avoids second-order gradients by using a proxy loss.
        
        Args:
            task_losses (List[Tensor]): scalar losses for each task, length T
            shared_params (Iterable[nn.Parameter]): parameters of shared layers
        
        Returns:
            total_loss (Tensor): weighted sum of task losses (for model update)
            gradnorm_loss (Tensor): proxy GradNorm loss (for weight update)
            normalized_weights (Tensor): normalized task weights (detached)
        """
        T = self.num_tasks
        weights = self.task_weights
        normalized_weights = weights / weights.mean()  # normalize to mean=1

        # Compute total loss for main model optimization
        total_loss = sum(normalized_weights[i] * task_losses[i] for i in range(T))

        # Compute gradient norms G_i = ||∇_θ (L_i)|| (note: not weighted by w_i here)
        # We use L_i directly (not w_i * L_i) to avoid coupling w_i into gradient computation
        G = []
        for i in range(T):
            assert task_losses[i].grad_fn is not None, f"Loss {i} has no grad_fn! Check if it's detached."
            # Use raw task loss (no weight) to compute gradient norm
            # This avoids needing create_graph=True and keeps G independent of weights
            grad = torch.autograd.grad(
                outputs=task_losses[i],
                inputs=shared_params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False
            )
            grad_flat = torch.cat([g.flatten() for g in grad if g is not None])
            G_i = torch.norm(grad_flat, p=2)
            G.append(G_i)
        G = torch.stack(G)  # shape: [T]

        # Compute target relative weights based on loss ratios and gradient norms
        with torch.no_grad():
            L = torch.stack(task_losses)
            L_avg = L.mean()
            # Relative inverse training rates
            r_target = (L / L_avg) ** self.alpha
            # Desired property: w_i ∝ r_target_i / G_i  => balance w_i * G_i across tasks
            target_weights_raw = r_target / (G + 1e-8)  # add epsilon for numerical stability
            target_weights = target_weights_raw / target_weights_raw.mean()  # normalize to mean=1

        # Proxy GradNorm loss: encourage current weights to match target weights
        # NOTE: normalized_weights requires grad; target_weights is constant
        gradnorm_loss = torch.abs(normalized_weights - target_weights).sum()

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
    Uncertainty weight strategy.
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
    Fixed uncertainty weight strategy.
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