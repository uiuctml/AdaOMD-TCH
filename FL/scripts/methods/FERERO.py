import json
import os
import time
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import ds_models
from torchvision.models import resnet18

# LR_DECAY = True
LR_DECAY = False


class FERERO(object):
    def __init__(self, config, train_client_loader, test_client_loader):
        self.config = config

        os.makedirs(self.config['output_dir'], exist_ok = True)
        self.log_fname = os.path.join(self.config['output_dir'], 'log.pickle')
        self.checkpoint_fname = os.path.join(self.config['output_dir'], 'checkpoint.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_client_loader = train_client_loader
        self.test_client_loader = test_client_loader
        
        self.preference = np.array([1/self.config['m']] * self.config['m'])
        self.setup_models()

        if 'w' in self.config:
            self.pref_vec = np.array(self.config['w']) / self.config['m']
        else:
            self.pref_vec = np.ones(self.config['m']) / self.config['m']
        self.init_lam_f = np.ones(self.config['m']) / self.config['m']
        self.init_lam_h = np.zeros(self.config['m'] - 1)
        self.init_lam = np.ones(self.config['m']) / self.config['m']
        self.iter_K = 1


    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        Model = getattr(ds_models, self.config['dataset'] + 'Full', None)
        if Model is None:
            raise ValueError(f"Unknown model name: {self.config['dataset']}Full")
        self.model = Model(self.config).to(self.device)
        # if self.config['dataset'] == 'CIFAR10':
        #     self.model = resnet18(num_classes=10).to(self.device)

        self.round = 0
        
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.config['checkpoint']:
            print('Loading checkpoint from', self.checkpoint_fname)
            states = torch.load(self.checkpoint_fname)
            state_dict = states['models']
            self.model.load_state_dict(state_dict)
            self.round = states['round']


    def run(self):
        logs = []

        for epoch in range(self.config['num_epochs']):
            self.round = epoch
            stats = {'round': self.round,
                     'lr': self.config['lr'],}
            
            t0 = time.time()
            self.train_one_round() # parameter updates
            stats['train_time'] = time.time() - t0

            stats['train_infer_stats'] = self.inference(train=True)
            stats['test_infer_stats'] = self.inference(train=False)

            self.print_stats(stats)
            logs.append(stats)
            if epoch % 10 == 0 or epoch == self.config['num_epochs'] - 1 :
                print('---------------- Saving ----------------')
                with open(self.log_fname, 'wb') as outfile:
                    pickle.dump(logs, outfile)
                    print(f'Logs written at {self.log_fname}')
                self.save_checkpoint()
                print(f'Checkpoint written at {self.checkpoint_fname}')
                print('----------------------------------------')
        
        print('--------------- All done ---------------')
        

    def print_stats(self, stats):
        print(f'[Round {self.round}]')
        print(f"Training set: mean_loss = {stats['train_infer_stats']['mean_loss']}; ",
              f"mean_acc = {stats['train_infer_stats']['mean_acc']}")
        print(f"Test set: mean_loss = {stats['test_infer_stats']['mean_loss']}; ",
              f"mean_acc = {stats['test_infer_stats']['mean_acc']}")


    def save_checkpoint(self):
        torch.save({'models': self.model.state_dict(), 'round': self.round}, self.checkpoint_fname)
   

    def inference(self, train=True):
        if self.config['dataset'] != 'CIFAR10':
            self.model.eval()
        t0= time.time()
        losses = []
        corrects, num_samples, accuracys = [], [], []

        with torch.no_grad(): 
            for m_i in range(self.config['m']):
                if train:
                    dataloader = self.train_client_loader[m_i]
                else:
                    dataloader = self.test_client_loader[m_i]
                
                # y_logit = self.model(X.to(self.device)).detach().cpu() # the global model
                # loss = self.criterion(y_logit, y).item()
                # num_batch = 1
                # n_correct = self.n_correct(y_logit, y)
                # num_sample = X.shape[0]
                loss, num_batch, n_correct, num_sample = 0., 0, 0, 0
                for (X, y) in dataloader:
                    y_logit = self.model(X.to(self.device)).detach().cpu() # the global model
                    loss += self.criterion(y_logit, y).item()
                    num_batch += 1
                    n_correct += self.n_correct(y_logit, y)
                    num_sample += X.shape[0]
                
                losses.append(loss / num_batch)
                corrects.append(n_correct)
                num_samples.append(num_sample)
                accuracys.append(n_correct / num_sample)
        
        # if train:
        #     print('Inference client training loss:', losses)
        #     print('Inference client training accuracy:', accuracys)
        loss = np.mean(losses)
        acc = np.sum(corrects) / np.sum(num_samples)
        infer_time= time.time() - t0
        infer_stats ={'mean_loss': loss, 'mean_acc': acc, 'infer_time': infer_time, 
                      'losses': losses, 'corrects': corrects , 'num_samples': num_samples, 'accuracys': accuracys}
        return infer_stats


    def train_one_round(self):
        self.model.train()
        updated_models = []
        client_losses= []

        for m_i in range(self.config['m']):
            dataloader = self.train_client_loader[m_i]
            model = copy.deepcopy(self.model)
            local_train_loss, num_batch = 0., 0
            for step_i in range(self.config['tau']):
                for (X, y) in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    model.zero_grad()
                    y_logit = model(X)
                    loss = self.criterion(y_logit, y)
                    loss.backward()
                    self.local_param_update(model, self.config['lr'])
                    local_train_loss += loss.detach().cpu().item()
                    num_batch += 1
            
            model.zero_grad()
            updated_models.append(model)
            local_train_loss = local_train_loss / float(num_batch) + 1e-10
            client_losses.append(local_train_loss)

        for client_model in updated_models:
            self.differentiate_learner(
                target_model = client_model,
                reference_state_dict = self.model.state_dict(),
                coeff = 1 / self.config['lr']
            )
        self.global_param_update(self.model, updated_models, client_losses, self.config['lr'])


    def local_param_update(self, model, lr):
        # gradient update manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad


    def differentiate_learner(self, target_model, reference_state_dict, coeff=1.):
        # set the gradients of the target model to be the difference between `target` and `reference` multiplied by `coeff`
        for name, param in target_model.named_parameters():
            if param.data.dtype == torch.float32:
                param.grad = coeff * (reference_state_dict[name].data - param.data)


    def global_param_update(self, global_model, local_models, client_losses, lr):
        grads = [0] * self.config['m']
        for m_i, local_model in enumerate(local_models):
            grads[m_i] = torch.cat([param.grad.data.clone().view(-1) for param in local_model.parameters() if param.grad is not None])
        grads = torch.vstack(grads)

        global_model.train()
        weighted_loss = 0.0
        for i in range(self.config['m']):
            loader = self.train_client_loader[i]
            X, y = next(iter(loader))
            X, y = X.to(self.device), y.to(self.device)
            loss_val = self.criterion(global_model(X), y)
            weighted_loss += self.init_lam[i] * loss_val
        global_model.zero_grad()
        weighted_loss.backward()
        grad_lamt = torch.cat([param.grad.data.clone().view(-1) for param in global_model.parameters() if param.grad is not None])

        try:
            weight_vec, nd, lam_f, lam_h = PMOLSolver.get_d_pmol(
                grads=grads.cpu().numpy(), 
                F=np.array(client_losses), 
                grad_lamt=grad_lamt.cpu().numpy(), 
                init_lam_f=self.init_lam_f, 
                init_lam_h=self.init_lam_h, 
                pref_vec=self.pref_vec, 
                iter_K=self.iter_K
            )
        except Exception as e:
            print("FERERO solver error:", e)
            weight_vec = self.init_lam_f

        weight_vec = np.array(weight_vec)
        weight_vec = weight_vec / weight_vec.sum()
        self.init_lam_f = lam_f
        self.init_lam_h = lam_h
        self.init_lam = weight_vec
        
        grads = {}
        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in grads:
                    grads[name] = torch.zeros_like(param.data)
                if param.grad is not None:
                    grads[name] += param.grad.clone() * weight_vec[m_i]

        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.data -= lr * grads[name]


    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit, 1)
        return (predicted == y).sum().item()


class PMOLSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = PMOLSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = PMOLSolver._min_norm_2d(vecs, dps)
        
        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < PMOLSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = PMOLSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = PMOLSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < PMOLSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
    
    def find_min_norm_element_PGD(grads):
        """
        We directly run the PGD until convergence
        """
        M = len(grads)
        # uniform as init
        init_lam = np.ones(M) / M
           
        iter_count = 0
        lam = init_lam 
        while iter_count < PMOLSolver.MAX_ITER:
            new_lam = lam - np.dot(grads, np.dot(grads.T, lam)) 
            new_lam = PMOLSolver._projection2simplex(new_lam)
            
            change = new_lam - lam
            if np.sum(np.abs(change)) < PMOLSolver.STOP_CRIT:
                break
            else:
                lam = new_lam
                iter_count += 1
                continue
        sol_lam = new_lam   
        nd = np.dot(grads.T, sol_lam)
        return sol_lam, nd
    
    def find_min_norm_element_PGD_H(grads, grad_lamt, 
                        init_lam_f, init_lam_h, 
                        A, Bh, H, iter_K):
        """
        We directly run the PGD with K iterations or till convergence
        with equality constraints H
        """
        M = len(grads)
        # uniform as init
        init_lam_f = np.ones(M) / M
           
        iter_count = 0

        lam_f = init_lam_f 
        Mh = Bh.shape[0]
        lam_h = init_lam_h
        gradsA = (grads.T @ A.T).T
        gradsBh = (grads.T @ Bh.T).T
        
        lam = np.dot(A.T, lam_f) + np.dot(Bh.T, lam_h)
        
        gamma = 1e-5
        while iter_count < iter_K:
            new_lam_f = lam_f - gamma * np.dot(gradsA, grad_lamt) 
            new_lam_f = PMOLSolver._projection2simplex(new_lam_f)
            
            new_lam_h = lam_h - gamma * (np.dot(
                gradsBh, grad_lamt) - 0.5 * H)            
            new_lam = np.dot(A.T, new_lam_f) + np.dot(Bh.T, new_lam_h)

            change = np.sum(np.abs(new_lam - lam))
            if change < PMOLSolver.STOP_CRIT:
                break
            else:
                lam_f = new_lam_f
                lam_h = new_lam_h
                lam = new_lam
                iter_count += 1
                continue
        
        nd_tplus = np.dot(grads.T, lam)
        return lam, nd_tplus, lam_f, lam_h
    
    def get_d_pmol(grads, F, grad_lamt, 
                init_lam_f, init_lam_h, 
                pref_vec, iter_K):
        """
        calculate the gradient direction for PMOL
        """
        nobj, dim = grads.shape
        if nobj <= 1:
            return np.array([1.])
        
        A = np.eye(nobj)
        # A = torch.from_numpy(A).float()
        if nobj == 2:
            Bh = np.array([pref_vec[1], -pref_vec[0]]).reshape((1, nobj))
        else:
            U, S, Vh = np.linalg.svd(pref_vec.reshape(1,-1), full_matrices=True)
            Bh = Vh[1:,:]
        # Bh = torch.from_numpy(Bh).float()
        H = Bh @ F
        sol, nd, lam_f, lam_h = PMOLSolver.find_min_norm_element_PGD_H(
                grads, grad_lamt, init_lam_f, init_lam_h, 
                A, Bh, H, iter_K)

        return sol, nd, lam_f, lam_h


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.dot(vecs[i][k], vecs[j][k]).item()#torch.dot(vecs[i][k], vecs[j][k]).data[0]
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.dot(vecs[i][k], vecs[i][k]).item()#torch.dot(vecs[i][k], vecs[i][k]).data[0]
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()#torch.dot(vecs[j][k], vecs[j][k]).data[0]
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        
        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum(
                [gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum(
                [gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))
