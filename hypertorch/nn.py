import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .math import *
import math


class RParameter(nn.Parameter):
    """
    A subclass for Riemannian parameter (to determine which parameters should be optimized by Riemannian optimizer).
    """
    pass


class SemiRiemannianModule(nn.Module):
    def e_parameters(self):
        for param in self.parameters():
            if not isinstance(param, RParameter):
                yield param

    def r_parameters(self):
        for param in self.parameters():
            if isinstance(param, RParameter):
                yield param


class MobiusLinear(SemiRiemannianModule):
    def __init__(self, in_features, out_features, bias=True, curv=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bias = RParameter(torch.empty((1, out_features))) if bias else None
        self.curv = curv
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = log_map0(x, self.curv)
        x = self.fc(x)
        x = exp_map0(x, self.curv)

        if self.bias is not None:
            x = mobius_addition(x, self.bias, self.curv)

        return x


class HyperbolicConcat(SemiRiemannianModule):
    def __init__(self, in_features, out_features, bias=True, curv=1, **kwargs):
        super().__init__()
        self.curv = curv
        self.mfc1 = MobiusLinear(in_features[0], out_features, False, curv=self.curv)
        self.mfc2 = MobiusLinear(in_features[1], out_features, False, curv=self.curv)
        self.bias = RParameter(torch.empty(1, out_features)) if bias else None
        if self.bias is not None:
            in_dim = sum(in_features)
            bound = 1 / math.sqrt(in_dim) if in_dim > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1, x2):
        x1 = self.mfc1(x1)
        x2 = self.mfc2(x2)
        x = mobius_addition(x1, x2, self.curv)

        if self.bias is not None:
            x = mobius_addition(x, self.bias, self.curv)

        return x


class MobiusReLU(SemiRiemannianModule):
    def __init__(self, curv):
        self.curv = curv
        super().__init__()

    def forward(self, x):
        x = mobius(F.relu, self.curv)(x)
        return x


class HyperbolicLinear(SemiRiemannianModule):
    """
    Wrapper for HyperbolicLinear with activation.
    """
    def __init__(self, input_dim, output_dim, activation, bias, curv):
        super().__init__()
        self.fc = MobiusLinear(input_dim, output_dim, bias, curv=curv)
        if activation == 'relu':
            self.activation = MobiusReLU(curv)
        elif activation == 'None':
            self.activation = lambda x: x
        else:
            raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')

    def forward(self, x):
        return self.activation(self.fc(x))


class HyperbolicLinearSkip(SemiRiemannianModule):
    """
    Wrapper for HyperbolicLinear with skip connection after activation.
    """
    def __init__(self, input_dim, output_dim, activation, bias, curv):
        super().__init__()
        self.fc = MobiusLinear(input_dim, output_dim, bias, curv=curv)
        if activation == 'relu':
            self.activation = MobiusReLU(curv)
        elif activation == 'None':
            self.activation = lambda x: x
        else:
            raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')

    def forward(self, x):
        y = self.fc(x)
        y = self.activation(y)
        return y + x


class HyperbolicFFN(nn.Sequential, SemiRiemannianModule):
    """
    Hyperbolic feedforward network.
    Optional skip connections (applies to all layers except last, for which user can choose).
    """

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 curv,
                 bias=True,
                 activation=None,
                 skips=False,
                 apply_to_last_layer=False,
                 **kwargs):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.curv = curv
        self.activation = activation
        self.bias = bias
        self.skips = skips
        self.apply_to_last_layer = apply_to_last_layer

        super().__init__(self.get_ffn_layers())

    def get_ffn_layers(self):
        layers = OrderedDict()

        def label_l(i):
            return f'HyperbolicLinear{i}'

        def label_ls(i):
            return f'HyperbolicLinearSkip{i}'

        prev = self.input_dim  # start with input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            if self.skips:
                layers[label_ls(i)] = HyperbolicLinearSkip(prev, hidden_dim, self.activation, self.bias, curv=self.curv)
            else:
                layers[label_l(i)] = HyperbolicLinear(prev, hidden_dim, self.activation, self.bias, curv=self.curv)

            prev = hidden_dim

        n = len(self.hidden_dims)
        if self.apply_to_last_layer:
            if self.skips:
                layers[label_ls(n)] = HyperbolicLinearSkip(prev, self.output_dim, self.activation, self.bias, curv=self.curv)
            else:
                layers[label_l(n)] = HyperbolicLinear(prev, self.output_dim, self.activation, self.bias, curv=self.curv)

        else:
            layers[label_l(n)] = HyperbolicLinear(prev, self.output_dim, self.activation, self.bias, curv=self.curv)

        return layers


class DoubleInputHyperbolicFFN(SemiRiemannianModule):
    """
    Hyperbolic feed-forward network with double input (suited to distance prediction).
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 curv,
                 **kwargs):
        super().__init__()
        input_dims = [input_dim] * 2
        if hidden_dims:
            ffn_input_dim = hidden_dims[0]
            ffn_hidden_dims = hidden_dims[1:]
            self.concat_layer = HyperbolicConcat(input_dims, ffn_input_dim, curv=curv, **kwargs)
            self.ffn = HyperbolicFFN(ffn_input_dim, ffn_hidden_dims, output_dim, curv=curv, **kwargs)
        else:
            self.concat_layer = HyperbolicConcat(input_dims, output_dim, curv=curv, **kwargs)
            self.ffn = lambda x: x

    def forward(self, x1, x2):
        x = self.concat_layer(x1, x2)
        x = self.ffn(x)
        return x





