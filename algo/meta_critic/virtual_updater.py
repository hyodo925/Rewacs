from collections import OrderedDict
import torch
class VirtualActorUpdater:
    def __init__(self):
        self.virtual_params = {}

    def step(self, model, grads, step_name, lr, from_params=None):
        if from_params is None:
            prev_params = OrderedDict(
                (name, param.data.clone()) for name, param in model.named_parameters()
            )
        else:
            prev_params = from_params

        new_params = OrderedDict()
        for (name, _), grad in zip(model.named_parameters(), grads):
            if grad is None:
                grad = torch.zeros_like(prev_params[name])
            if grad.shape != prev_params[name].shape:
                raise RuntimeError(f"Shape mismatch at {name}: param={prev_params[name].shape}, grad={grad.shape}")
            new_params[name] = prev_params[name] - lr * grad

        self.virtual_params[step_name] = new_params

    def get(self, step_name):
        return self.virtual_params[step_name]