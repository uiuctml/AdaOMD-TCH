import numpy as np
import torch

def allocation_rotation(ds_list, config):

    np.random.seed(config['data_seed'])

    res = []
    for ds_id, ds in enumerate(ds_list):
        (X, y) = ds
        m = config['m']
        classes2indx = build_class_dict(y)

        client_data_id = [[] for _ in range(m)]
        for key in classes2indx.keys():
            if ds_id == 0 and config['data_mode'] == 'less':
                per_class_ll = list(np.random.permutation(classes2indx[key][:len(classes2indx[key])//10]))
            else:
                per_class_ll = list(np.random.permutation(classes2indx[key]))
            per_class_ll2 = chunkify(per_class_ll, m)
            client_data_id = [client_data_id[m_i]+per_class_ll2[m_i] for m_i in range(m)]
        
        client_data = []
        for m_i in range(m):
            p_i = config['cluster_ids'][m_i]
            if config['p'] == 2:
                p_i = p_i * 2
            X_batch = X[client_data_id[m_i]]
            y_batch = y[client_data_id[m_i]]
            if config['dataset'] == 'MNIST':
                X_batch2 = torch.rot90(X_batch, k=p_i, dims=(1,2))
                X_batch2 = X_batch2.reshape(X_batch2.shape[0], -1) # flatten
            elif config['dataset'] == 'CIFAR10':
                X_batch2 = torch.rot90(X_batch, k=p_i, dims=(2,3))
            client_data.append((X_batch2, y_batch))

        res.append(client_data)

    return res


def build_class_dict(y):
    classes2indx = {}
    for ind, label in enumerate(y):
        label =int(label)
        if label in classes2indx:
            classes2indx[label].append(ind)
        else:
            classes2indx[label] = [ind]
    return classes2indx


def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    # gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    gen = (a[i * k : (i + 1) * k] for i in range(n))
    return list(gen)