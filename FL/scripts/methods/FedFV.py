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

import ds_models

# LR_DECAY = True
LR_DECAY = False


class FedFV(object):

    def __init__(self, config, train_client_loader, test_client_loader):
        self.config = config

        os.makedirs(self.config['output_dir'], exist_ok = True)
        self.log_fname = os.path.join(self.config['output_dir'], 'log.pickle')
        self.checkpoint_fname = os.path.join(self.config['output_dir'], 'checkpoint.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_client_loader = train_client_loader
        self.test_client_loader = test_client_loader
        
        self.setup_models()

        self.alpha =  self.config['alpha']
        self.fedfv_tau =  self.config['fedfv_tau']
        self.client_last_sample_round = [-1] * self.config['m']
        self.client_grads_history = [0] * self.config['m']


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
        return infer_stats    


    def train_one_round(self):
        self.model.train()
        gradients = []
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
            local_train_loss = local_train_loss / float(num_batch) + 1e-10
            client_losses.append(local_train_loss)
            gradients.append(compute_grad_update(old_model=self.model, new_model=model)) # list of tensors

        grads= [flatten(gradient) for gradient in gradients]    # compute pseudo-gradient 
        # update GH
        for cid, gi in zip(list(range(self.config['m'])), grads):
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = self.round
        
        # project grads
        order_grads = copy.deepcopy(grads)
        order = [_ for _ in range(len(order_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(client_losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]
        # keep the original direction for clients with the αm largest losses
        keep_original = []
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]

        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = grads[j].dot(order_grads[i])
                    if dot < 0:
                        order_grads[i] = order_grads[i] - grads[j] * dot / (norm(grads[j])**2)

        gt = self._model_average(order_grads)
        # mitigate external conflicts
        if self.round >= self.fedfv_tau:
            for k in range(self.fedfv_tau-1, -1, -1):
                # calculate outside conflicts
                gcs = [self.client_grads_history[cid] for cid in range(self.config['m']) if self.client_last_sample_round[cid] == self.round - k and gt.dot(self.client_grads_history[cid]) < 0]
                if gcs:
                    g_con = self._model_sum(gcs)
                    dot = gt.dot(g_con)
                    if dot < 0:
                        gt = gt - g_con*dot/(norm(g_con)**2)

        # ||gt||=||1/m*Σgi||
        gnorm = norm(self._model_average(grads))
        gt = gt/norm(gt)*gnorm
        unflat_gt =  unflatten(gt, gradients[0])
        add_update_to_model(self.model, unflat_gt)


    def _model_average(self, mds=[], weights = []):
        
        if len(mds)==0:
            return None
      
        md_avg= torch.zeros(mds[0].shape).to(self.device)
        if len(weights) == 0: weights = [1.0 / len(mds) for _ in range(len(mds))]
        for wid in range(len(mds)):
            weight = weights[wid]
            md_avg= md_avg + mds[wid] * weight
        return md_avg
    

    def _model_sum(self, mds=[]):
        
        if len(mds)==0:
            return None
      
        md_avg= torch.zeros(mds[0].shape).to(self.device)
        
        for wid in range(len(mds)):
            
            md_avg= md_avg + mds[wid] 
        return md_avg


    def local_param_update(self, model, lr):
        # gradient update manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad


    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit, 1)
        correct = (predicted == y).sum().item()
        return correct


def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
		flattened = flattened[n_params:]
	return grad_update


def compute_grad_update(old_model, new_model):
    return [(new_param.data - old_param.data) 
            for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_update_to_model(model, update, weight=1.0):
	if not update: return model
	
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))