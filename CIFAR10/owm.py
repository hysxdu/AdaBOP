import sys, time
import numpy as np
import torch

dtype = torch.cuda.FloatTensor  # run on GPU

import utils

import torch.nn.functional as F



########################################################################################################################

class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=64, lr=0,  clipgrad=10, args=None):
        self.model = model
        self.pre_h = None
        self.pre_x = None
        self.loss = None
        self.prototype_class={}

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad


        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        with torch.no_grad():
            self.Pc1 = torch.autograd.Variable(torch.eye(3 * 4 * 4).type(dtype))
            self.Pc2 = torch.autograd.Variable(torch.eye(64 * 3 * 3).type(dtype))
            self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype))
            self.P1 = torch.autograd.Variable(torch.eye(256 * 4 * 4).type(dtype))
            self.P2 = torch.autograd.Variable(torch.eye(1024).type(dtype))
            self.P3 = torch.autograd.Variable(torch.eye(1024).type(dtype))

        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm}
                                     ], lr=lr, momentum=0.9)

        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data):

        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr)
        nepochs = self.nepochs
        self.loss = 0
        test_max = 0
        loss_his=[]

        # Loop epochs
        try:
            for e in range(nepochs):



                self.train_epoch(xtrain, ytrain, cur_epoch=e, nepoch=nepochs,task_id=t)

                #if (e+1) % 5  !=0:
                 #   continue
                train_loss, train_acc = self.eval(xtrain, ytrain)
                loss_his.append(train_loss)
                print('| [{:d}/10], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                # # Valid
                valid_loss, valid_acc = self.eval(xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()
               # if (e+1)%5!=0&t!=9:
                  # continue
                xtest = data[10]['test']['x'].cuda()
                ytest = data[10]['test']['y'].cuda()

                _, test_acc = self.eval(xtest, ytest)

                # # Adapt lr
                # if valid_loss < best_loss:
                #     best_loss = min(best_loss,valid_loss)

                # if valid_acc > best_acc:
                #     best_acc = max(best_acc, valid_acc)
                if test_acc>self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                best_model = utils.get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return


    def get_space_cnn_0(self,x_list,w,stride):
        feature_space = []
        x_list_0 = x_list[0]
        _, _, H, W = x_list_0.shape
        F, _, HH, WW = w.shape
        S = stride  # stride
        Ho = int(1 + (H - HH) / S)
        Wo = int(1 + (W - WW) / S)
        l = len(x_list)
        for i in range(Ho):
            for j in range(Wo):
                fea = x_list_0[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                cov_matrix = torch.mm(torch.t(fea),fea)
        w_t = w.view(F, -1) 
        return cov_matrix
    
    def get_space_cnn_1(self,x_list,w,stride):
        feature_space = []
        x_list_1 = x_list[1]
        _, _, H, W = x_list_1.shape
        F, _, HH, WW = w.shape
        S = stride  # stride
        Ho = int(1 + (H - HH) / S)
        Wo = int(1 + (W - WW) / S)
        l = len(x_list)
        for i in range(Ho):
            for j in range(Wo):
                fea = x_list_1[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                cov_matrix = torch.mm(torch.t(fea),fea) 
        w_t = w.view(F, -1) 
        return cov_matrix

    def get_space_cnn_2(self,x_list,w,stride):
        feature_space = []
        x_list_2 = x_list[2]
        _, _, H, W = x_list_2.shape
        F, _, HH, WW = w.shape
        S = stride  # stride
        Ho = int(1 + (H - HH) / S)
        Wo = int(1 + (W - WW) / S)
        l = len(x_list)
        for i in range(Ho):
            for j in range(Wo):
                fea = x_list_2[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                cov_matrix = torch.mm(torch.t(fea),fea)
        w_t = w.view(F, -1) 
        return cov_matrix

    def get_space(self, h_list):
        feature_space = []
        r = h_list[0].size(1)
        l = len(h_list)
        for i in range(l):
            fea = h_list[i]
            cov_matrix = torch.mm(torch.t(fea),fea)
            feature_space.append(cov_matrix)
        return feature_space

    def get_trust(self, w, cov):
        grad = w.grad.data.detach()
        proj = torch.mm(grad, cov)
        alpha = torch.norm(grad) / torch.norm(proj)
        #print(alpha)
        if alpha >  0.2:
            eta = 0.54
        else:
            eta = 0.56
        return eta    
    
    def get_trust_cnn(self,w,cov):
        F, _, _, _ = w.shape

        grad = w.grad.data.view(F, -1).detach()
        proj = torch.mm(grad, cov)
        alpha = torch.norm(grad) / torch.norm(proj)
        #print(alpha)
        if alpha >  0.2:
            eta = 0.54
        else:
            eta = 0.56
        return eta    




    def train_epoch(self, x, y, cur_epoch=0, nepoch=0,task_id=0):
        self.model.train()

        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):

            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
           # print("a")
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                targets = torch.autograd.Variable(y[b])
          #  images1 = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])  # 4B
           # images2 = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])  # 4B
            # images1 = torch.cat([Rotation(x, r) for r in range(4)])
            # images2 = torch.cat([Rotation(x, r) for r in range(4)])

         #   images_pair = torch.cat([images, self.simclr_aug(images)], dim=0)  # 8B

            labels1 = targets
            # print("LLLL",labels1.shape)
            #rot_sim_labels = torch.cat([labels1 + 10 * i for i in range(4)], dim=0)
            # print("LLLL",labels1.shape)

           # Rot_sim_labels = torch.cat([labels1 + 0 * i for i in range(oop)], dim=0)
            #      rot_sim_labels1 = torch.cat([torch.arange(0,labels1.shape[0]).cuda()+ 10*i for i in range(4)], dim=0)
            # print(labels1)
            # print(Rot_sim_labels)
            # print("RRRR1",rot_sim_labels)da()  # 这个label其实是用来mask的，4个rotate
          #  rot_sim_labels = rot_sim_labels
            #  images_pair = simclr_aug(images_pair)  # simclr augment
            # print("X",input_x.shape,images_pair.shape)
            #  for n,w in Basic_model.named_parameters():
            #     print(n,w.shape)
     #       outputs_aux = self.model.conv_feature(images_pair)#, is_simclr=True)
            # outputs_aux1 = Basic_model(images_pair, is_simclr1=True)# , penultimate=True)

     #       simclr = normalize(outputs_aux)  # normalize
            # simclr1 = normalize(outputs_aux1)
            # print(images1.shape[0],simclr.shape)
          #  f1, f2 = torch.split(simclr, [images.shape[0], images.shape[0]], dim=0)
          #  features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # print(f1.shape,f2.shape,features.shape)
            # if opt.method == 'SupCon':
          #  criterion = SupConLoss(temperature=0.05)
           # loss_sup1 = criterion(features, rot_sim_labels)
      #      sim_matrix = get_similarity_matrix(simclr)
            # sim_matrix1 = get_similarity_matrix(simclr1)
            # print("1")
      #      loss_sim1 = Supervised_NT_xent(sim_matrix, labels=labels1,
   #                                        temperature=0.07)

            # Forward
           # output1, h_list1, x_list1 = self.model.forward(self.simclr_aug(images))
          #  print("b")
          #  output1, h_list1, x_list1 = self.model.forward(self.simclr_aug(images))
            feature,x_list,num = self.model.conv_feature(images)
            output, h_list = self.model.forward(feature,num)
            eta = []
            eta_cnn = []
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()

            loss.backward()

            if i_batch != 0:
                for n, w in self.model.named_parameters():
                    if n == 'c1.weight':
                        eta_cnn_1 = self.get_trust_cnn(w,cov_cnn_0)
                        eta_cnn.append(eta_cnn_1)
                    if n == 'c2.weight':
                        eta_cnn_2 = self.get_trust_cnn(w,cov_cnn_1)
                        eta_cnn.append(eta_cnn_2)
                    if n == 'c3.weight':
                        eta_cnn_3 = self.get_trust_cnn(w,cov_cnn_2)
                        eta_cnn.append(eta_cnn_3)
                    if n == 'fc1.weight':
                        eta1 = self.get_trust(w,cov[0])
                        #print('eta1',eta1)
                        eta.append(eta1)
                    if n == 'fc2.weight':
                        eta2 = self.get_trust(w,cov[1])
                        #print('eta2',eta2)
                        eta.append(eta2)
                    if n == 'fc3.weight':
                        eta3 = self.get_trust(w,cov[2])
                        #print('eta3',eta3)
                        eta.append(eta3)
            cov = self.get_space(h_list)

            for n, w in self.model.named_parameters():
                if n == 'c1.weight':
                    cov_cnn_0 = self.get_space_cnn_0(x_list, w, stride=2)
                if n == 'c2.weight':
                    cov_cnn_1 = self.get_space_cnn_1(x_list, w, stride=2)
                if n == 'c3.weight':
                    cov_cnn_2 = self.get_space_cnn_2(x_list, w, stride=2)    


            
            lamda = i_batch / len(r_len)/nepoch + cur_epoch/nepoch
            

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            ''' 
            mean_h_1 = torch.mean(torch.matmul(h_list[0].t(), h_list[0]))
            mean_h_2 = torch.mean(torch.matmul(h_list[1].t(), h_list[1]))
            mean_h_3 = torch.mean(torch.matmul(h_list[2].t(), h_list[2]))
            std_v_1 = torch.var(h_list[0])
            std_v_2 = torch.var(h_list[1])
            std_v_3 = torch.var(h_list[2])
            alpha_array[1] = (mean_h_1.data.detach() ** 2 + std_v_1.detach()) ** lamda
            alpha_array[2] = (mean_h_2.data.detach() ** 2 + std_v_2.detach()) ** lamda
            alpha_array[3] = (mean_h_3.data.detach() ** 2 + std_v_3.detach()) ** lamda
            '''


            def pro_weight(p, x, w,alpha=1.0, cnn=True, stride=1):
                lamda = 0.5
                if cnn:
                    F, _, HH, WW = w.shape
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            #print(x.size())
                            #m = int(x.shape[2] * 0.9)
                            #x_svd = TSVD(x,m)
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p.data, torch.t(r))
                            p.sub_(lamda * torch.mm(k, torch.t(k)) / (alpha + lamda * torch.mm(r, k)))

                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)

                else:
                    #m = int(x.shape[0] * 0.9)
                    #x_SVD = SVD(x,m)
                    #r = x_SVD.data
                    r = x.data
                    k = torch.mm(p.data, torch.t(r))
                    p.sub_(lamda * torch.mm(k, torch.t(k)) / (alpha + lamda * torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

            def pro_weight_trust(p, x, w, eta, alpha=1.0, cnn=True, stride=1):
                #if cur_epoch<=3:
                if cnn:
                    F, _, HH, WW = w.shape
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p.data, torch.t(r))
                            p.sub_(0.6 * torch.mm(k, torch.t(k)) / (alpha + 0.6 * torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)

                else:

                    r = x.data
                    k = torch.mm(p.data, torch.t(r))
                    p.sub_(0.6 * torch.mm(k, torch.t(k)) / (alpha + 0.6 * torch.mm(r, k)))

                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

            if i_batch == 0:
                for n, w in self.model.named_parameters():
                    if n == 'c1.weight':
                        pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                    if n == 'c2.weight':
                        pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                    if n == 'c3.weight':
                        pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                    if n == 'fc1.weight':
                        pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                    if n == 'fc2.weight':
                        pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)

                    if n == 'fc3.weight':
                        pro_weight(self.P3,  h_list[2], w, alpha=alpha_array[3], cnn=False)
            else:
                for n, w in self.model.named_parameters():
                    if n == 'c1.weight':
                        pro_weight_trust(self.Pc1, x_list[0], w, eta_cnn[0], alpha=alpha_array[0], stride=2)

                    if n == 'c2.weight':
                        pro_weight_trust(self.Pc2, x_list[1], w, eta_cnn[1], alpha=alpha_array[0], stride=2)

                    if n == 'c3.weight':
                        pro_weight_trust(self.Pc3, x_list[2], w, eta_cnn[2], alpha=alpha_array[0], stride=2)

                    if n == 'fc1.weight':
                        pro_weight_trust(self.P1,  h_list[0], w, eta[0], alpha=alpha_array[1], cnn=False)

                    if n == 'fc2.weight':
                        pro_weight_trust(self.P2,  h_list[1], w, eta[1], alpha=alpha_array[2], cnn=False)

                    if n == 'fc3.weight':
                        pro_weight_trust(self.P3,  h_list[2], w, eta[2], alpha=alpha_array[3], cnn=False)
            # Compensate embedding gradients

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                targets = torch.autograd.Variable(y[b])

            # Forward
            feature,x_list,num = self.model.conv_feature(images)
            output,  _ = self.model.forward(feature,num)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
           # print(pred,targets)
            hits = (pred % 10 == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num
    
    