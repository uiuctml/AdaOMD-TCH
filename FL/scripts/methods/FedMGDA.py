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

from torch.linalg import norm
import math
import quadprog

import ds_models

# LR_DECAY = True
LR_DECAY = False


class FedMGDA(object):

    def __init__(self, config, train_client_loader, test_client_loader):
        self.config = config

        os.makedirs(self.config['output_dir'], exist_ok = True)
        self.log_fname = os.path.join(self.config['output_dir'], 'log.pickle')
        self.checkpoint_fname = os.path.join(self.config['output_dir'], 'checkpoint.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_client_loader = train_client_loader
        self.test_client_loader = test_client_loader
        
        self.setup_models()

        self.epsilon =  self.config['epsilon']
        self.lambda_star = [0] * self.config['m']


    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        Model = getattr(ds_models, self.config['dataset'] + 'Full', None)
        if Model is None:
            raise ValueError(f"Unknown model name: {self.config['dataset']}Full")
        self.model = Model(self.config).to(self.device)

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
        t0 = time.time()
        losses = []
        corrects, num_samples, accuracys = [], [], []

        with torch.no_grad(): 
            for m_i in range(self.config['m']):
                if train:
                    dataloader = self.train_client_loader[m_i]
                else:
                    dataloader = self.test_client_loader[m_i]
                
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
        
        loss = np.mean(losses)
        acc = np.sum(corrects) / np.sum(num_samples)
        infer_time= time.time() - t0
        infer_stats ={'mean_loss': loss, 'mean_acc': acc, 'infer_time': infer_time, 
                      'losses': losses, 'corrects': corrects , 'num_samples': num_samples, 'accuracys': accuracys}
        if train:
            infer_stats['lambda_star'] = self.lambda_star
        return infer_stats    


    def train_one_round(self):
        self.model.train()
        gradients = []
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
            gradients.append(compute_grad_update(old_model=self.model, new_model=model)) # list of tensors

        grads= [flatten(gradient) for gradient in gradients]    # compute pseudo-gradient
        norms = [norm(g) for g in grads] # compute pseudo-grad norms
        normalized_grads= [g/n for g,n in zip(grads,norms)] # normalized pseudo-gradients

        self.lambda_star = list(solve_centered_w(normalized_grads, epsilon=self.epsilon))
        
        self.global_param_update(self.model, updated_models)


    def local_param_update(self, model, lr):
        # gradient update manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad


    def global_param_update(self, global_model, local_models):
        weights = {}
        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)
                weights[name] += param.data * self.lambda_star[m_i]

        for name, param in global_model.named_parameters():
            param.data = weights[name]


    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit, 1)
        correct = (predicted == y).sum().item()
        return correct


def solve_centered_w(U, epsilon):
    """
        utils from FedMGDA repo
        U is a list of normalized gradients (stored as state_dict()) from n users
    """
    n = len(U)
    K = np.eye(n,dtype=float)
    for i in range(n):
        for j in range(n):
            K[i,j] = torch.dot(U[i], U[j]).cpu().numpy()

    Q = 0.5 *(K + K.T)
    p = np.zeros(n,dtype=float)
    a = np.ones(n,dtype=float).reshape(-1,1)
    Id = np.eye(n,dtype=float)
    neg_Id = -1. * np.eye(n,dtype=float)
    lower_b = (1./n - epsilon) * np.ones(n,dtype=float)
    upper_b = (-1./n - epsilon) * np.ones(n,dtype=float)
    A = np.concatenate((a,Id,Id,neg_Id),axis=1)
    b = np.zeros(n+1)
    b[0] = 1.
    b_concat = np.concatenate((b,lower_b,upper_b))
    alpha = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
    #print('weights of FedMGDA: ', alpha)
    return alpha


def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])


def compute_grad_update(old_model, new_model):
    return [(new_param.data - old_param.data)
            for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))