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

# LR_DECAY = True
LR_DECAY = False


class AFLeg(object):

    def __init__(self, config, train_client_loader, test_client_loader):
        self.config = config

        os.makedirs(self.config['output_dir'], exist_ok = True)
        self.log_fname = os.path.join(self.config['output_dir'], 'log.pickle')
        self.checkpoint_fname = os.path.join(self.config['output_dir'], 'checkpoint.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_client_loader = train_client_loader
        self.test_client_loader = test_client_loader
        
        self.setup_models()

        self.dynamic_lambdas = torch.ones(self.config['m']) * 1.0 / self.config['m']
        self.init_losses = [0 for _ in range(self.config['m'])]


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
            self.dynamic_lambdas = torch.tensor(states['dynamic_lambdas'])
            self.round = states['round']

        self.result_model = copy.deepcopy(self.model)


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
        torch.save({'models': self.result_model.state_dict(), 'dynamic_lambdas': self.dynamic_lambdas,
                    'round': self.round}, self.checkpoint_fname)
   

    def inference(self, train=True):
        if self.config['dataset'] != 'CIFAR10':
            self.result_model.eval()
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
                    y_logit = self.result_model(X.to(self.device)).detach().cpu() # the global model
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
            infer_stats['dynamic_lambdas'] = self.dynamic_lambdas
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
            if self.config['ratio'] == 1:
                if self.init_losses[m_i] == 0:
                    self.init_losses[m_i] = local_train_loss
                local_train_loss = local_train_loss / self.init_losses[m_i]
            client_losses.append(local_train_loss)

        for client_model in updated_models:
            self.differentiate_learner(
                target_model = client_model,
                reference_state_dict = self.model.state_dict(),
                coeff = 1 / self.config['lr']
            )
        self.global_param_update(self.model, updated_models, self.config['lr'])
        
        self.dynamic_lambdas_update(client_losses, self.config['learning_rate_lambda'])
        
        self.result_model_update(self.result_model, self.model.state_dict())


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


    def global_param_update(self, global_model, local_models, lr):
        # average of each weight
        aggregation_weights =  self.dynamic_lambdas / torch.sum(self.dynamic_lambdas)
        
        grads = {}
        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in grads:
                    grads[name] = torch.zeros_like(param.data)
                if param.grad is not None:
                    if 'w' in self.config:
                        grads[name] += param.grad.clone() * aggregation_weights[m_i] * self.config['w'][m_i]
                    else:
                        grads[name] += param.grad.clone() * aggregation_weights[m_i]

        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.data -= lr * grads[name]


    def dynamic_lambdas_update(self, client_losses, lr):
        if 'w' in self.config:
            self.dynamic_lambdas = [lmb_i * np.exp(lr * w_i * loss_i) for lmb_i, w_i, loss_i in zip(self.dynamic_lambdas, self.config['w'], client_losses)]
        else:
            self.dynamic_lambdas = [lmb_i * np.exp(lr * loss_i) for lmb_i, loss_i in zip(self.dynamic_lambdas, client_losses)]
        self.dynamic_lambdas = torch.tensor(self.dynamic_lambdas) / sum(self.dynamic_lambdas)


    def result_model_update(self, result_model, model_state_dict):
        for name, param in result_model.named_parameters():
            param.data = (param.data * self.round + model_state_dict[name].data * 1) / (self.round + 1)


    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit, 1)
        correct = (predicted == y).sum().item()
        return correct


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))