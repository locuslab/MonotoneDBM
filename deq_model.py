import torch
import torch.nn as nn
import torch.nn.functional as F
from multitier_conv import *
from proxsoftmax import *
from util import *
import numpy as np

class ConvDeqCrf(nn.Module):
    def __init__(self, splittingMethod, sizes, kernels, data_shape, MON_DEFAULTS, m=0.1, **kwargs):
        super().__init__()
        self.linear_module = MultiMonotoneHollowConv(sizes, kernels, monotone_m=m)
        self.nonlin_module = prox_softmax(alpha=MON_DEFAULTS['alpha'], dim=1)
        self.tau = nn.Parameter(torch.full(data_shape, np.sqrt(10.)))
        self.clftau = nn.Parameter(torch.tensor([1.]))
        self.mon = splittingMethod(self.linear_module, self.nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x, mask=0):
        self.linear_module.clean_norms()
        z = self.mon(x, mask=mask)
        out = self.linear_module.tensor_to_tuple(z)
        return out

def run_tune_alpha(model, x, max_alpha, mask=0):
    print("----tuning alpha----")
    print("current: ", model.mon.alpha)
    orig_alpha = model.mon.alpha
    model.mon.alpha = max_alpha
    model.nonlin_module.alpha = model.mon.alpha

    with torch.no_grad():
        model((x,), mask=mask)
    iters = model.mon.forward_steps
    iters_n = iters
    print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
    while model.mon.alpha > 1e-4 and iters_n <= iters:
        model.mon.alpha = model.mon.alpha/2
        model.nonlin_module.alpha = model.mon.alpha
        with torch.no_grad():
            model((x,), mask=mask)
        iters = iters_n
        iters_n = model.mon.forward_steps
        print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))

    if iters==model.mon.max_iter:
        print("none converged, resetting to current")
        model.mon.alpha=orig_alpha
        model.nonlin_module.alpha = model.mon.alpha
    else:
        model.mon.alpha = model.mon.alpha * 2
        model.nonlin_module.alpha = model.mon.alpha
        print("setting to: ", model.mon.alpha)
    print("--------------\n")


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0, mask=0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, length = x0.shape
    X = torch.zeros(bsz, m, length, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, length, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.contiguous().view(bsz, -1), f(x0).contiguous().view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res

class MONForwardBackwardSplitting(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=0.9, tol=1e-5, max_iter=50, solver=anderson):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        self.forward_steps = 0
        self.backward_steps = 0
        self.forward_res = 0
        self.backward_res = 0
        self.f = lambda z, injection, mask: self.damped_forward(self.alpha, mask, injection, z)

    def damped_forward(self, alpha, mask, injection, z):
        z_tuple = self.linear_module.tensor_to_tuple(z)
        masked_z = z_tuple
        masked_z[0] = masked_z[0] * (1 - mask) + mask * injection[0]
        if len(injection) > 1:
            masked_z[-1] = injection[-1][:, :, None, None]
        linear_out_tuple = self.linear_module(*masked_z)
        linear_out = tuple(alpha * linear_out_tuple[i] for i in range(len(z_tuple)))
        out = []
        for i in range(len(linear_out)):
            num_group = self.linear_module.groups[i]
            curr_lin_out = (1 - alpha) * z_tuple[i] + linear_out[i]
            bsz, c, h, w = curr_lin_out.shape
            curr_out = self.nonlin_module(curr_lin_out.view(-1, c // num_group, h, w)).view(bsz, c, h, w)
            out.append(curr_out)
        return self.linear_module.tuple_to_tensor(out)

    def forward(self, x, mask=0):

        """ Forward pass of the MON, find an equilibirum with forward-backward splitting
            x is the input injection, mask is an indicator matrix that tells which parts are observed
        """

        # Run the forward pass _without_ tracking gradients
        z_start_tuple = [F.softmax(torch.zeros((x[0].shape[0], *shape), dtype=x[0].dtype, device=x[0].device), 1) for
                         shape in self.linear_module.shapes]
        z_start = self.linear_module.tuple_to_tensor(z_start_tuple)

        with torch.no_grad():
            results = self.solver(lambda z: self.f(z, x, mask),
                                  z_start, max_iter=self.max_iter, tol=self.tol, mask=mask)

        z = results[0]
        errs = results[1]
        self.forward_steps = len(errs)
        self.forward_res = errs[-1]

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.f(z, x, mask)
        z0 = zn.clone().detach().requires_grad_()
        f0 = self.f(z0, x, mask)

        def backward_hook(grad):
            g, errs = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                  grad, max_iter=self.max_iter, tol=self.tol)

            self.backward_steps = len(errs)
            self.backward_res = errs[-1]
            return g

        if zn.requires_grad:
            zn.register_hook(backward_hook)

        self.errs = errs
        return zn
