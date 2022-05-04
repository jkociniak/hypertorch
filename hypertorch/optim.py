import torch
from torch.optim import Optimizer
from .math import conformal_factor, exp_map


class RiemannianSGD(Optimizer):
    """
    Implements Riemannian SGD.
    """
    def __init__(self, params, lr=1e-3):
        assert lr > 0
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    riemannian_grad = grad.mul(1/(conformal_factor(p) ** 2))
                    new_p = exp_map(p, -lr * riemannian_grad)
                    p.data.copy_(new_p)

        return loss

