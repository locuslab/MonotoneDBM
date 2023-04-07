import torch
import torch.nn as nn

class StableLogW(torch.autograd.Function):
    """ This computes the log of the function:
    f(y) = (alpha)/(1-alpha) * W(e^(y/alpha - 1) * (1-alpha)/alpha)
    i.e., this function outputs g(y) = log(f(y))
    However, trying to compute this function directly from the Lambert W function
    (and then taking a log) wouldn't be stable, so we compute the function directly.

    This function is relevant to our setting because it gives the solution to the
    Lagrangian of the softmax prox operator, for different alpha, i.e., the solution
    of the prox_(-H(x)-||x||^2/2)(x) is to find nu such that such that
    \sum_i f(x_i + nu) = 1

    By the definition of the function above, letting x = g(y), one can show that this
    is equivalent to finding a root of the equation in x,
    h(x) = -alpha * x + (y - alpha) - (1-alpha)e^x == 0
    We find this root using Halley's method
    which requires the first two derivatives of h
    h'(x) = -alpha - (1-alpha)e^X
    h''(x) = -(1-alpha)e^X
    """

    @staticmethod
    def forward(ctx, y, alpha):
        tol = 1e-4
        max_iter = 5000
        it = 0
        ctx.alpha = alpha

        x = torch.min(y / alpha - 1, y - 1)  # initial guess, reasonable over [-inf, 1]
        while True:
            exp_x = torch.exp(x)
            f = -alpha * x + (y - alpha) - (1 - alpha) * exp_x
            if (f.abs().max() <= tol or it == max_iter):
                ctx.save_for_backward(y, x)
                return x

            df = -alpha - (1 - alpha) * exp_x
            ddf = -(1 - alpha) * exp_x
            x = x - (2 * f * df) / (2 * (df ** 2) - f * ddf)
            if torch.isnan(x).sum()>0:
                assert False
            it += 1
    @staticmethod
    def backward(ctx, grad_out):
        y, g = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            return grad_out / (y - ctx.alpha * g), None
        else:
            return None, None


class ProxSoftmax(torch.autograd.Function):
    """ This computes the proximal operator
    x = prox^alpha_f(y)
    where f = -H(y) - ||y||^2/2
    over the dimension dim.  Note that this reduces to softmax for alpha=1.

    We compute this, as higlighted above, by finding the nu such that
    \sum_i exp(g(x_i + nu)) = 1
    for the above function g.  We find this solution using Newton's method over nu.
    Because the above function is convex monotonic in nu over [-inf,1], Newton's method
    with step size = 1 will always converge.

    Backprop is done via computing d nu* / d y via implicit function theorem.
    """

    @staticmethod
    def forward(ctx, y, alpha, dim):
        tol = 1e-4
        ctx.alpha = alpha
        ctx.dim = dim
        max_iter = 5000
        it = 0
        nu = -y.max(dim=dim, keepdim=True)[0] + 1
        while True:
            l = StableLogW.apply(y + nu, alpha)
            p = torch.exp(l)
            f = torch.sum(p, dim=dim, keepdim=True) - 1
            if f.abs().max() < tol or it == max_iter:
                ctx.save_for_backward(y, p, nu)
                return p
            df = torch.sum(p / (y + nu - alpha * l), dim=dim, keepdim=True)
            nu = nu - f / df
            it += 1
    @staticmethod
    def backward(ctx, grad_out):
        y, p, nu = ctx.saved_tensors
        l = StableLogW.apply(y + nu, ctx.alpha)  # makes double backprop "work"

        if ctx.needs_input_grad[0]:
            g = p / (y + nu - ctx.alpha * l)
            dnu = torch.sum(grad_out * g, dim=ctx.dim, keepdim=True) * (g / torch.sum(g, dim=ctx.dim, keepdim=True))
            return grad_out * g - dnu, None, None
        else:
            return None, None, None

class prox_softmax(nn.Module):
    def __init__(self, alpha, dim):
        super().__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, x):
        return ProxSoftmax.apply(x, self.alpha, self.dim)