# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.optim import Optimizer


class QHM(Optimizer):
    r"""
    Stochastic gradient method with Quasi-Hyperbolic Momentum (QHM):

        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k)
        x(k+1) = x(k) - \alpha * d(k)

    "Quasi-hyperbolic momentum and Adam for deep learning"
        by Jerry Ma and Denis Yarats, ICLR 2019

    optimizer = QHM(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0)

    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range[0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range[0,1] (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum \beta and dampened gradient (1-\beta)
            \nu = \beta: SGD with "Nesterov momentum" \beta
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = torch.optim.QHM(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantee of convergence?
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        super(QHM, self).__init__(params, defaults)

        # extra_buffer == True only in SSLS with momentum > 0 and nu != 1
        self.state['allocate_step_buffer'] = False

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.qhm_direction()
        self.qhm_update()

        return loss

    def add_weight_decay(self):
        # weight_decay is the same as adding L2 regularization
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay > 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

    def qhm_direction(self):

        for group in self.param_groups:
            momentum = group['momentum']
            qhm_nu = group['qhm_nu']

            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data  # Optimization parameters
                g = p.grad.data  # Stochastic gradient

                # Compute the (negative) step directoin d and necessary momentum
                state = self.state[p]
                if abs(momentum) < 1e-12 or abs(qhm_nu) < 1e-12:  # simply SGD if beta=0 or nu=0
                    d = state['step_buffer'] = g
                else:
                    if 'momentum_buffer' not in state:
                        h = state['momentum_buffer'] = torch.zeros_like(x)
                    else:
                        h = state['momentum_buffer']
                    # Update momentum buffer: h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
                    h.mul_(momentum).add_(g, alpha=1 - momentum)

                    if abs(qhm_nu - 1) < 1e-12:  # if nu=1, then same as SGD with momentum
                        d = state['step_buffer'] = h
                    else:
                        if self.state['allocate_step_buffer']:  # copy from gradient
                            if 'step_buffer' not in state:
                                state['step_buffer'] = torch.zeros_like(g)
                            d = state['step_buffer'].copy_(g)
                        else:  # use gradient buffer
                            d = state['step_buffer'] = g
                        # Compute QHM momentum: d(k) = (1 - \nu) * g(k) + \nu * h(k)
                        d.mul_(1 - qhm_nu).add_(h, alpha=qhm_nu)

    def qhm_update(self):
        """
        Perform QHM update, need to call compute_qhm_direction() before calling this.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(self.state[p]['step_buffer'], alpha=-group['lr'])

