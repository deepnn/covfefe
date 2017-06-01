from __future__ import absolute_import
from __future__ import print_function

import torch.optim as nn

def adadelta(w, lr=1.0, rho=0.9, eps=1e-06, w_decay=0):
    return nn.Adadelta(params=w, lr=lr,
                       rho=rho, eps=eps,
                       weight_decay=w_decay)

def adagrad(w, lr=0.01, lr_decay=0, w_decay=0):
    return nn.Adagrad(params=w, lr=lr,
                      lr_decay=lr_decay,
                      weight_decay=w_decay)

def adam(w, lr=0.001, betas=(0.9, 0.999), eps=1e-08, w_decay=0):
    return nn.Adam(params=w, lr=lr,
                   betas=betas, eps=eps,
                   weight_decay=w_decay)

def adamax(w, lr=0.002, betas=(0.9, 0.999), eps=1e-08, w_decay=0):
    return nn.Adamax(params=w, lr=lr,
                     betas=betas, eps=eps,
                     weight_decay=w_decay)

def asgd(w, lr=0.01, lam=0.0001, alpha=0.75, t0=1000000.0, w_decay=0):
    return nn.ASGD(params=w, lr=lr,
                   lambd=lam, alpha=alpha,
                   t0=t0, weight_decay=w_decay)

def lbfgs(w, lr=1, max_iter=20, max_eval=None, tol_grad=1e-05,
          tol_change=1e-09, hist_size=100, line_search_fun=None):
    return nn.LBFGS(params=w, lr=lr, max_iter=max_iter,
                    max_eval=max_eval, tolerance_grad=tol_grad,
                    tolerance_change=tol_change, history_size=hist_size,
                    line_search_fn=line_search_fun)

def rms_prop(w, lr=0.01, etas=(0.5,1.2), step_sz=(1e-06, 50)):
    return nn.RMSprop(params=w, lr=lr, etas=etas,
                      step_sizes=step_sz)

def r_prop(w, lr=0.01, etas=(0.5, 1.2), step_sz=(1e-06, 50)):
    return nn.Rprop(params=w, lr=lr, etas=etas,
                    step_sizes=step_sz)

def sgd(w, lr=0.1, m=0, damp=0, w_decay=0, nesterov=False):
    return nn.SGD(params=w, lr=lr, momentum=m,
                  dampening=damp, weight_decay=w_decay,
                  nesterov=nesterov)
