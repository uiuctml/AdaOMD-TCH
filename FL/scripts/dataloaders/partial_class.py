import numpy as np
import torch

def allocation_partial_class(ds_list, config):

    np.random.seed(config['data_seed'])

    class_keys = np.unique(ds_list[0][1])
    client_class_keys = []
    for m_i in range(config['m']):
        if config['n_i'] != 0:
            n_i = config['n_i']
        else:
            n_i = np.random.randint(1, len(class_keys)+1)
        keys_i = np.random.choice(class_keys, size=n_i, replace=False)
        client_class_keys.append(sorted(keys_i))

    res = []
    for ds_id, ds in enumerate(ds_list):
        (X, y) = ds

        m = config['m']
        classes2indx = build_class_dict(y)

        class_chunkified_ids = {}
        for key in classes2indx.keys():
            if ds_id == 0 and config['data_mode'] == 'less':
                class_data_ids = list(np.random.permutation(classes2indx[key][:len(classes2indx[key])//10]))
            else:
                class_data_ids = list(np.random.permutation(classes2indx[key]))
            chunkified_ids = chunkify(class_data_ids, m)
            class_chunkified_ids[key] = chunkified_ids
        
        client_data = []
        for m_i in range(m):
            client_data_id = []
            for key in client_class_keys[m_i]:
                client_data_id += class_chunkified_ids[key][m_i]
            X_batch = X[client_data_id]
            y_batch = y[client_data_id]
            if config['dataset'] == 'MNIST':
                X_batch = X_batch.reshape(X_batch.shape[0], -1) # flatten
            client_data.append((X_batch, y_batch))
        
        res.append(client_data)
    
    res.append(client_class_keys)
    return res


def build_class_dict(y):
    classes2indx = {}
    for ind, label in enumerate(y):
        label = int(label)
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