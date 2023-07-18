#2020-07-31
import math
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer
from sklearn.decomposition import PCA
import numpy as np

class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, owm=False,
                 weight_decay=0, amsgrad=False):
        # p: last p columns of eigen-vector
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, owm=owm,
                        )
        self.proj = None
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('owm', False)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            owm = group['owm']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                update = self.get_update(group, grad, p)
                if owm and self.proj is not None:
                    if len(update.shape) == 4:
                        update_ = torch.mm(update.view(update.size(0), -1), self.proj[p]).view_as(update)
                    else:
                        update_ = torch.mm(update, self.proj[p])

                else:
                    update_ = update
                p.data.add_(update_)
        return loss

    # def get_transforms(self, Proj, fea_in, alpha):
    #     for group in self.param_groups:
    #         owm = group['owm']
    #         if owm is False:
    #             continue
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             proj = Proj[p]
    #             if proj is False:
    #                 if len(p.shape) == 4:
    #                     _, in_c, h, w = p.size()
    #                     size = in_c * h * w
    #                 elif len(p.shape) == 2:
    #                     h, w = p.size()
    #                     size = h * w
    #                 else:
    #                     raise ValueError('only conv2d and fully connected layer')
    #                 proj = torch.eye(size, dtype=p.dtype, device=p.device)
    #
    #             k = torch.mm(proj, torch.t(fea_in[p]))
    #             eye = torch.eye(fea_in[p].size(0), dtype=p.dtype, device=p.device)
    #             inv = alpha * eye + torch.mm(fea_in[p], k)
    #             inv = torch.inverse(inv)
    #             sub = torch.mm(torch.mm(k, inv), torch.t(k))
    #             proj.sub_(sub)
    #             Proj[p] = proj
    #     return Proj

    def SVD(self,P,k):
        U,sigma,VT = torch.linalg.svd(P)
        sigma1 = torch.diag(sigma)
        U = U[:,:k]
        sigma_SVD = sigma1[:k,:k]
        VT = VT[:k,:]
        svd1 = torch.mm(U,sigma_SVD)
        svd = torch.mm(svd1,VT)
        return svd

    def TSVD(self,P,k):
        U,sigma,VT = torch.linalg.svd(P)
        #sigma1 = torch.diag(sigma)
        U1 = U[:,:k]
        #sigma_SVD = sigma1[:k,:k] / (torch.norm(sigma1[:k,:k]))
        #VT = VT[:k,:]
        #svd1 = torch.mm(U1,sigma_SVD)
        #svd = torch.mm(svd1,VT)
        svd = torch.mm(U1,torch.t(U1))
        return svd


    def PCA_svd(self, X, k, center=True):
        cov = torch.mm(torch.t(X),X)
        U,sigma,VT = torch.linalg.svd(cov)
        sigma_diag =  torch.diag(sigma)
        sigma_k = sigma_diag[:k, :k]
        VT_k = VT[:k, :]
        svd = torch.mm(sigma_k, VT_k)
        return svd

    def get_trust(self,fea_in,thres = 0.2):
        for group in self.param_groups:
            for p in group['params']:
                cov = torch.mm(fea_in[p].t(),fea_in[p])
                grad = p.grad.data
                proj = torch.mm(grad,cov)
                grad_norm = torch.norm(grad)
                proj_norm = torch.norm(proj)
                if proj_norm > (thres * grad_norm):
                    trust_num = 1
                else:
                    trust_num = 0
        return  trust_num

    def get_transforms(self, Proj, fea_in, alpha, train_name):
        for group in self.param_groups:
            owm = group['owm']
            if owm is False:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                proj = Proj[p]
                if proj is False:
                    if len(p.shape) == 4:
                        _, in_c, h, w = p.size()
                        size = in_c * h * w
                    elif len(p.shape) == 2:
                        h, w = p.size()
                        size = h * w
                    else:
                        raise ValueError('only conv2d and fully connected layer')
                    proj = torch.eye(size, dtype=p.dtype, device=p.device)
                
                
                if train_name == '1':
                    cov = torch.mm(fea_in[p].t(),fea_in[p])
                    adpt = 0.5
                else:
                    if len(p.shape) == 4:
                        grad = p.grad.data.view(p.size(0),-1)
                        grad_proj = torch.mm(grad,cov)
                        grad_norm = torch.norm(grad)
                        proj_norm = torch.norm(grad_proj)
                        adpt = proj_norm / grad_norm
                    else:
                        grad = p.grad.data
                        grad_proj = torch.mm(grad,cov)
                        grad_norm = torch.norm(grad)
                        proj_norm = torch.norm(grad_proj)
                        adpt = proj_norm / grad_norm
                        cov = torch.mm(fea_in[p].t(),fea_in[p])
            
                l = fea_in[p].size(0)
                '''
                m = fea_in[p].size(1)
                if l < m:
                    fea_in[p]=self.SVD(fea_in[p],int(0.9 * l))
                else:
                    fea_in[p]=self.SVD(fea_in[p],int(0.9 * m))
                '''    
                for i in range(l):
                    fea = fea_in[p][i].unsqueeze(0).detach()
                    #print('fea:',fea.size())
                    k = torch.mm(proj, torch.t(fea))
                    proj.sub_(torch.mm(k, torch.t(k)) / (adpt * alpha + torch.mm(fea, k)))
                Proj[p] = proj
        return Proj

    def set_proj(self, proj):
        self.proj = proj

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update


