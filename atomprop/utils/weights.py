"""
Module for weight stratergies among pretraining tasks.
"""
import torch

class WeightStratergy:
    """
    Base class for weight stratergy.
    """
    @staticmethod
    def _weight_norm(weights):
        """
        Normalize the weights to sum to 1.
        """
        return weights / torch.sum(weights)
    
    def __init__(self, task_num, device, input_weights = None):
        self.task_num = task_num
        self.device = device
        self.input_weights = torch.ones(task_num, device=device) if not input_weights else input_weights.to(device)
    
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
    def __init__(self, task_num, device, input_weights = None):
        super().__init__(task_num, device, input_weights)
    
    def outputs(self):
        return self.input_weights
    
class HardSwitch(WeightStratergy):
    """
    Hard switch weight stratergy.
    """
    def __init__(self, task_num, device, switch_timings = None, input_weights = None):
        super().__init__(task_num, device, input_weights)
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
            return self._weight_norm(self.input_weights)
        else:
            weights = torch.zeros(self.task_num, device=self.device)
            weights[current_task] = self.input_weights[current_task]
            return weights
        
class SoftSwitch(WeightStratergy):
    def __init__(self, task_num, device, switch_timings=None, input_weights=None, transition_width=10, 
                 min_weight=0.0, max_weight=1.0):
        super().__init__(task_num, device, input_weights)
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
        
        return self._weight_norm(weights * self.input_weights)

class GradNorm(WeightStratergy):
    """
    GradNorm weight stratergy.
    Reference: https://arxiv.org/abs/1705.07115
    """
    def __init__(self, task_num, device, input_weights = None):
        super().__init__(task_num, device, input_weights)
        
    def outputs(self, grads: list):
        grads_norm = []
        for grad in grads:
            if grad is None:
                grads_norm.append(torch.tensor(0.0, device=self.device))
            else:
                grads_norm.append(grad.norm())
        norms = torch.stack(grads_norm)
        self.inv = 1.0 / (norms.detach() + 1e-8)
        return self._weight_norm(self.inv * self.input_weights)