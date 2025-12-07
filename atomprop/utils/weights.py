"""
Module for weight stratergies among pretraining tasks.
"""
import torch
from atomprop.utils.grads_utils import find_optimal_weights

class WeightStratergy:
    """
    Base class for weight stratergy.
    """
    @staticmethod
    def weight_norm(weights, eps=1e-3):
        """
        Normalize the weights to sum to 1.
        """
        return weights / torch.sum(weights)
    
    def __init__(self, task_num, device):
        self.task_num = task_num
        self.device = device
    
    def outputs(self):
        """
        Get the weights for each task.
        Return a pytorch tensor of shape (task_num,).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def outputs_to_list(self, **kwargs):
        """
        Get the weights for each task as a list.
        Return a list of length task_num.
        """
        weights = self.outputs(**kwargs)
        return torch.unbind(weights, dim=0)
    
class EqualWeightStratergy(WeightStratergy):
    """
    Equal weight stratergy.
    """
    def __init__(self, task_num, device):
        super().__init__(task_num, device)
    
    def outputs(self, divide=False):
        if not divide:
            return torch.ones(self.task_num).to(self.device)
        return torch.ones(self.task_num)/self.task_num.to(self.device)
    
class HardSwitch(WeightStratergy):
    """
    Hard switch weight stratergy.
    """
    def __init__(self, task_num, device, switch_timings = None):
        super().__init__(task_num, device)
        assert switch_timings is not None, "switch_timings must be provided for HardSwitchWeightStratergy."
        assert switch_timings.shape[0] == task_num, "switch_timings length must be task_num."
        self.switch_timings = switch_timings.to(device)
    
    def outputs(self, timing):
        current_task = -1
        for i in range(self.task_num):
            if timing >= self.switch_timings[i]:
                current_task = i
                break
        if current_task == -1:
            return torch.ones(self.task_num)
        else:
            weights = torch.zeros(self.task_num, device=self.device)
            weights[current_task] = torch.tensor(1.0, dtype=torch.float32)
            return weights
        
class SoftSwitch(WeightStratergy):
    def __init__(self, task_num, device, switch_timings=None, transition_width=10, 
                 min_weight=0.0, max_weight=1.0):
        super().__init__(task_num, device)
        assert switch_timings is not None, "switch_timings must be provided for SoftSwitchWeightStratergy."
        assert switch_timings.shape[0] == task_num, "switch_timings length must be task_num."
        self.switch_timings = switch_timings.to(device)
        self.transition_width = transition_width
        self.min_weight = min_weight
        self.max_weight = max_weight
        assert max_weight > min_weight, "max_weight must be greater than min_weight"
    
    def outputs(self, timing):
        weights = torch.full((self.task_num,), self.min_weight, device=self.device)
        weight_range = self.max_weight - self.min_weight
        
        # Determine which task is currently active or transitioning
        if timing < self.switch_timings[0]:
            # First task is active
            weights[0] = self.max_weight
        else:
            for i in range(1, self.task_num):
                if timing < self.switch_timings[i]:
                    # Task i is transitioning to active
                    prev_timing = self.switch_timings[i-1]
                    if timing < prev_timing + self.transition_width:
                        # Transition period: task i-1 -> task i
                        progress = (timing - prev_timing) / self.transition_width
                        cos_value = torch.cos(torch.pi * progress)
                        weights[i-1] = self.min_weight + 0.5 * weight_range * (1 + cos_value)
                        weights[i] = self.min_weight + 0.5 * weight_range * (1 - cos_value)
                    else:
                        # Task i is fully active
                        weights[i] = self.max_weight
                    break
            else:
                # Last task is active
                if timing < self.switch_timings[-1] + self.transition_width:
                    # Transition to last task
                    progress = (timing - self.switch_timings[-1]) / self.transition_width
                    cos_value = torch.cos(torch.pi * progress)
                    weights[-2] = self.min_weight + 0.5 * weight_range * (1 + cos_value)
                    weights[-1] = self.min_weight + 0.5 * weight_range * (1 - cos_value)
                else:
                    # Last task fully active
                    weights[-1] = self.max_weight
        
        return weights

class GradNorm(WeightStratergy):
    """
    GradNorm weight stratergy.
    Reference: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, task_num, device):
        super().__init__(task_num, device)
        
    def outputs(self, grads: list):
        grads_norm = []
        for grad in grads:
            if grad is None:
                grads_norm.append(torch.tensor(0.0, device=self.device))
            else:
                grads_norm.append(grad.norm())
        norms = torch.stack(grads_norm)
        self.inv = 1.0 / (norms.detach() + 1e-8)
        return self.inv
    
class ParetoOpt(WeightStratergy):
    """
    Given a group of grads g1, g2, ..., gn, this method calculates w1, w2, ..., wn so that g_opt = Σwi*gi satisfies g_opt = argmin max ∠(g_opt, gi).
    """
    
    def __init__(self, task_num, device):
        super().__init__(task_num, device) 
    
    def outputs(self, grads: list):
        grads_mean = []
        for grad in grads:
            grads_mean.append(torch.mean(grad, dim=0))
        ws = find_optimal_weights(grads_mean)
        return ws.to(self.device)