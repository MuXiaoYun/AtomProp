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
    """
    Soft switch weight stratergy.
    It's similar to hard switch. The difference is that it uses a cosine function to smooth the transition between tasks.
    """
    def __init__(self, task_num, device, switch_timings = None, input_weights = None, transition_width = 10):
        super().__init__(task_num, device, input_weights)
        assert switch_timings is not None, "switch_timings must be provided for SoftSwitchWeightStratergy."
        assert switch_timings.shape[0] == task_num, "switch_timings length must be task_num."
        self.switch_timings = switch_timings.to(device)
        self.transition_width = transition_width
    
    def outputs(self, timing):
        weights = torch.zeros(self.task_num, device=self.device)
        for i in range(self.task_num):
            if i == 0:
                if timing < self.switch_timings[i]:
                    weights[i] = 1.0
                elif timing < self.switch_timings[i] + self.transition_width:
                    weights[i] = 0.5 * (1 + torch.cos(torch.pi * (timing - self.switch_timings[i]) / self.transition_width))
            else:
                if timing >= self.switch_timings[i-1] + self.transition_width and timing < self.switch_timings[i]:
                    weights[i] = 1.0
                elif timing >= self.switch_timings[i] and timing < self.switch_timings[i] + self.transition_width:
                    weights[i] = 0.5 * (1 + torch.cos(torch.pi * (timing - self.switch_timings[i]) / self.transition_width))
        return self._weight_norm(weights*self.input_weights)

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