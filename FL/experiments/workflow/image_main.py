import argparse
import yaml
import sys

import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch

import dataloaders
import methods


def get_config():

    config = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--data-mode", type=str, default="full")
    parser.add_argument("--allocation", type=str, default="rotation")
    parser.add_argument("--preference", type=int, default=0)
    parser.add_argument("--method", type=str, default="FedAvg")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output/MNIST_rotation/FedAvg")
    parser.add_argument("--checkpoint", type=int, default=0)

    parser.add_argument("--gamma", type=float, default=1.) # STche
    parser.add_argument("--momentum", type=int, default=0) # STche
    parser.add_argument("--learning-rate-lambda", type=float, default=0.01) # AFL, AFL_new, AFLeg, AFLeg_new
    parser.add_argument("--q", type=float, default=0.) # qFFL
    parser.add_argument("--epsilon", type=float, default=1.) # FedMGDA
    parser.add_argument("--alpha", type=float, default=0.1) # FedFV
    parser.add_argument("--fedfv-tau", type=int, default=0) # FedFV
    parser.add_argument("--base", type=float, default=5.) # PropFair

    parser.add_argument("--n-i", type=int, default=0) # number of classes per client in partial_class

    parser.add_argument("--ratio", type=int, default=0) # FedAvg, Tche, STche, AFL, AFL_new, AFLeg, AFLeg_new
    parser.add_argument("--iterate-eval", type=str, default='mini-batch') # AFL_new, AFLeg_new

    args = parser.parse_args()
    config.update(vars(args))

    suffix = f'_m{args.preference}' if args.preference != 0 else ''
    with open(f"configs/{args.dataset}_{args.allocation}{suffix}.yaml", "r") as read_file:
        config.update(yaml.safe_load(read_file))
    
    return config


def main():

    config = get_config()
    print('--------------- Settings ---------------')
    print('[Dataset]', config['dataset'])
    print('[Allocation scheme]', config['allocation'])
    if config['allocation'] == 'partial_class':
        print('[n_i]', config['n_i'])
    print('[Method]', config['method'])
    print('[Data seed]', config['data_seed'])
    print('[Train seed]', config['train_seed'])
    if 'AFL' in config['method']:
        print('[learning rate lambda]', config['learning_rate_lambda'])
    elif config['method'] == 'STche':
        print('[gamma]', config['gamma'])
        print('[momentum]', 'True' if config['momentum'] else 'False')
    elif config['method'] == 'qFFL':
        print('[q]', config['q'])
    elif config['method'] == 'FedMGDA':
        print('[epsilon]', config['epsilon'])
    elif config['method'] == 'FedFV':
        print('[alpha]', config['alpha'])
        print('[tau]', config['fedfv_tau'])
    elif config['method'] == 'PropFair':
        print('[base]', config['base'])
    print('----------------------------------------')

    train_data, test_data = load_dataset(config['dataset'])

    allocation_scheme = getattr(dataloaders, 'allocation_' + config['allocation'], None)
    if allocation_scheme is None:
        raise ValueError(f"Unknown allocation scheme: {config['allocation']}")
    client_data = allocation_scheme([train_data, test_data], config)

    if config['dataset'] == 'MNIST':
        train_client_loader = [[(X, y)] for (X, y) in client_data[0]]
        test_client_loader = [[(X, y)] for (X, y) in client_data[1]]
        # train_client_loader, test_client_loader = client_data[0], client_data[1]
    elif config['dataset'] == 'CIFAR10':
        train_client_loader = [DataLoader(LocalDS(X, y), batch_size=config['batch_size'], shuffle=True)
                               for (X, y) in client_data[0]]
        if config['iterate_eval'] != 'val':
            test_client_loader = [DataLoader(LocalDS(X, y), batch_size=config['batch_size'], shuffle=True)
                                for (X, y) in client_data[1]]
            val_client_loader = None
        else: # this is only implemented for CIFAR_rotation
            client_data_val, client_data_test = [], []
            for (X, y) in client_data[1]:
                # print(y)
                g = torch.Generator(device=X.device)
                g.manual_seed(config['data_seed'])
                pos = torch.randperm(100, generator=g, device=X.device)[:50]
                offsets = torch.arange(10) * 100
                idx = (offsets[:, None] + pos[None, :]).reshape(-1)
                X_val, y_val = X[idx], y[idx]
                client_data_val.append((X_val, y_val))
                # print(y_val)
                mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
                mask[idx] = False
                idx_comp = mask.nonzero(as_tuple=True)[0]
                X_test, y_test = X[idx_comp], y[idx_comp]
                client_data_test.append((X_test, y_test))
            val_client_loader = [DataLoader(LocalDS(X, y), batch_size=100, shuffle=True)
                                for (X, y) in client_data_val]
            test_client_loader = [DataLoader(LocalDS(X, y), batch_size=config['batch_size'], shuffle=True)
                                for (X, y) in client_data_test]

    MethodClass = getattr(methods, config['method'], None)
    if MethodClass is None:
        raise ValueError(f"Unknown method: {config['method']}")
    if config['method'] in ['AFL_new', 'AFLeg_new']:
        exp = MethodClass(config, train_client_loader, test_client_loader, val_client_loader)
    else:
        exp = MethodClass(config, train_client_loader, test_client_loader)

    exp.run()


def load_dataset(ds_name):
    
    res = []

    for train in [True, False]:

        if ds_name == 'MNIST':
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ds = datasets.MNIST(root='./data', train=train, download=True, transform=transforms)
        elif ds_name == 'Fashion':
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ds = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transforms)
        elif ds_name == 'CIFAR10':
            transforms = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(32, padding=4),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor()
                        ])
            ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=transforms)
            
        dl = DataLoader(ds)
        X, y = dl.dataset.data, dl.dataset.targets  # (60000,28,28) for MNIST, (50000,32,32,3) for CIFAR10
        if ds_name == 'CIFAR10':
            X = torch.tensor(X).permute(0,3,1,2)
            y = torch.tensor(y)
        # normalize to have 0 ~ 1 range in each pixel
        X = X / 255.0

        res.append((X, y))
    
    return res


class LocalDS(Dataset):
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    main()