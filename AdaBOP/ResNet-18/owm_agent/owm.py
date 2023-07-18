from .agent import *
import re
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class OWMAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.fea_in_hook = {}
        self.modules_in_hook = {}
        self.init_model_optimizer()
        self.projection = defaultdict(bool)

        self.empFI = True
        self.regularization_terms = {}
        self.reg_params = {n: p for n, p in self.model.named_parameters() if 'bn' in n}

    def hook(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
            # self.fea_in_hook.update({module.weight: torch.mean(fea_in[0], 0, True)})
            self.fea_in_hook.update({module.weight: fea_in[0]})
        elif isinstance(module, nn.Conv2d):
            fea_in_ = []
            kernel_size = module.kernel_size
            stride = module.stride
            if module.padding_mode != 'zeros':
                fea_in_padding = F.pad(fea_in[0], module._reversed_padding_repeated_twice, mode=module.padding_mode)
            else:
                fea_in_padding = F.pad(fea_in[0], module._reversed_padding_repeated_twice)

            # fea_in_padding = torch.mean(fea_in_padding, 0, True)
            _, _, h, w = fea_in_padding.shape
            ho = int(1 + (h - kernel_size[0]) / stride[0])
            wo = int(1 + (w - kernel_size[1]) / stride[1])
            fea_dim = module.in_channels * kernel_size[0] * kernel_size[1]
            for i in range(ho):
                for j in range(wo):
                    r = fea_in_padding[:, :,
                        i * stride[0] : i * stride[0] + kernel_size[0],
                        j * stride[1] : j * stride[1] + kernel_size[1]].contiguous().view(-1, fea_dim)
                    fea_in_.append(r)
            fea_in_ = torch.cat(fea_in_, dim=0)
            self.fea_in_hook.update({module.weight: fea_in_})
            # self.fea_in_hook.update({module.weight: fea_in[0]})
        return None

    def init_model_optimizer(self):
        fea_params = [p for n, p in self.model.named_parameters() if not bool(re.match('last', n)) and 'bn' not in n]
        cls_params = [p for n, p in self.model.named_parameters() if bool(re.match('last', n))]
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
        model_optimizer_arg = {'params': [{'params': fea_params, 'owm': True, 'lr': self.config['owm_lr']},
                                          {'params': cls_params, 'weight_decay': 0.0, 'lr': self.config['head_lr']},
                                          {'params': bn_params, 'lr': self.config['bn_lr']}],
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config['model_optimizer'] is 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.5)

    def train_task(self, train_loader, train_name, val_loader=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)

        if self.task_count < self.num_task:

            if self.reset_model_optimizer:  # Reset model optimizer before learning each task
                self.log('Classifier Optimizer is reset!')
                self.init_model_optimizer()
                self.model.zero_grad()
            with torch.no_grad():
                self.log('update transforms')
                self.update_optim_transforms(train_loader,train_name)
                self.log('done')
                self.model_optimizer.set_proj(self.projection)

            self.task_count += 1
            if self.reg_params:
                task_param = {}
                for n, p in self.reg_params.items():
                    task_param[n] = p.clone().detach()
                importance = self.calculate_importance(train_loader)
                self.regularization_terms[self.task_count] = {'importance': importance, 'task_param': task_param}
    
    
    
    
    def update_optim_transforms(self, train_loader,train_name):
        modules = [m for n, m in self.model.named_modules() if hasattr(m, 'weight') and not bool(re.match('last', n))]
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.hook))
        for i, (inputs, target, task) in tqdm(enumerate(train_loader),
                                              desc='update transforms',
                                              total=len(train_loader)):
            if self.config['gpu']:
                inputs = inputs.cuda()
            self.model.forward(inputs)
            alpha = 0.05
            self.projection=self.model_optimizer.get_transforms(self.projection, self.fea_in_hook, alpha, train_name)

        for h in handles:
            h.remove()
        self.fea_in_hook = {}
        torch.cuda.empty_cache()

    def calculate_importance(self, dataloader):
        self.log('computing EWC')
        importance = {}
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)


        mode = self.model.training
        self.model.eval()
        for _, (inputs, targets, task) in tqdm(enumerate(dataloader),
                                               desc='computing importance',
                                               total=len(dataloader)):
            if self.config['gpu']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = self.model.forward(inputs)

            if self.empFI:
                ind = targets
            else:
                task_name = task[0] if self.multihead else 'ALL'
                pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
                                                                                         :self.valid_out_dim]
                ind = pred.max(1)[1].flatten()

            l = self.criterion(output, ind, task, regularization=False)
            self.model.zero_grad()
            l.backward()

            for n, p in importance.items():
                if self.reg_params[n].grad is not None:
                    p += ((self.reg_params[n].grad ** 2) * len(inputs) / len(dataloader))

        self.model.train(mode=mode)
        return importance

    def reg_loss(self):
        self.reg_step += 1
        reg_loss = 0
        for i, reg_term in self.regularization_terms.items():
            task_reg_loss = 0
            importance = reg_term['importance']
            task_param = reg_term['task_param']
            for n, p in self.reg_params.items():
                task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
            reg_loss += task_reg_loss
            self.summarywritter.add_scalar('reg_loss/task_%d' % i, task_reg_loss, self.reg_step)
        return reg_loss





def owm_based(config):
    return OWMAgent(config)
