import torch
import numpy as np

TOLERANCE_EPS = 1e-7
STABILITY_EPS = 1e-6


def validate(x):  # ensure that x is not too close to 0
    zeros = torch.zeros_like(x)
    mask = torch.isclose(x, zeros, atol=TOLERANCE_EPS, rtol=0)
    zeros[mask] = STABILITY_EPS
    return x + zeros


def mobius_addition(x, y, c):
    dot_xy = torch.sum(x * y, dim=1, keepdim=True)
    dot_xx = torch.sum(x * x, dim=1, keepdim=True)
    dot_yy = torch.sum(y * y, dim=1, keepdim=True)
    numerator = (1 + 2 * c * dot_xy + c * dot_yy) * x + (1 - c * dot_xx) * y
    denominator = 1 + 2 * c * dot_xy + c * c * dot_xx * dot_yy
    return numerator / denominator


def mobius_addition_np(x, y, c):
    dot_xy = np.sum(x * y)
    dot_xx = np.sum(x * x)
    dot_yy = np.sum(y * y)
    numerator = (1 + 2 * c * dot_xy + c * dot_yy) * x + (1 - c * dot_xx) * y
    denominator = 1 + 2 * c * dot_xy + c * c * dot_xx * dot_yy
    return numerator / denominator


def exp_map(x, v, c):
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    cf = conformal_factor(x, c)
    scalar_factor = torch.tanh(torch.sqrt(c) * cf * v_norm / 2) / (torch.sqrt(c) * v_norm)
    return mobius_addition(x, scalar_factor * v, c)


def exp_map0(v, c):
    v = validate(v)
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    scalar_factor = torch.tanh(torch.sqrt(c) * v_norm) / (torch.sqrt(c) * v_norm)
    return scalar_factor * v


def log_map(x, y, c):
    v = mobius_addition(-x, y, c)
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    cf = conformal_factor(x, c)
    scalar_factor = 2 * torch.arctanh(torch.sqrt(c) * v_norm) / (torch.sqrt(c) * cf * v_norm)
    return scalar_factor * v


def log_map0(y, c):
    y = validate(y)
    y_norm = torch.linalg.norm(y, dim=1, keepdim=True)
    scalar_factor = torch.arctanh(torch.sqrt(c) * y_norm) / (torch.sqrt(c) * y_norm)
    return scalar_factor * y


def conformal_factor(x, c):
    return 2 / (1 - c * torch.linalg.norm(x) ** 2)


def hyperbolic_dist(x):
    norm = torch.linalg.norm(x, dim=1)
    res = 2 * torch.arctanh(norm)
    return res


def mobius(f, c):
    return lambda x: exp_map0(f(log_map0(x, c)), c)


def h2p(x):
    t = x[:, 0]
    x = x[:, 1:]
    return x / (1 + t)


def p2h(y):
    t = (1 + (y**2).sum(dim=1)) / (1 - (y**2).sum(dim=1))
    x = 2*y / (1 - (y**2).sum(dim=1))
    return torch.cat([t, x], dim=1)


def minkowski_inner_product(x, y):
    x_h, y_h = p2h(x), p2h(y)
    xy_h = x_h * y_h
    prod = xy_h[:, 0] - xy_h[:, 1:].sum(dim=1)
    return prod

