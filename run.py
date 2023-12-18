import anndata as ad
import hdf5plugin
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
import math
import scanpy as sc
import wandb
import argparse
from tqdm import tqdm
from copy import deepcopy
from CellPLM.utils.eval import minimum_eval, clustering_eval, CountCorr
from CellPLM.utils.data import XDict, clean_batches, balanced_partition, data_setup
from CellPLM.model import OmicsFormer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process, Manager
import pickle
import os
import json
import random
torch.set_num_threads(16)
import torch 
from torch import nn



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def info_nce_loss(features, temperature=0.1):

    labels = torch.cat([torch.arange(features.shape[0] //2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    criterion = torch.nn.CrossEntropyLoss()    
    logits = logits / temperature
    
    return criterion(logits, labels)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.set_sharing_strategy('file_system')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config, shared_data, tune_flag, dataset_list, gene_dict, partitions, val_num):
    # rank, world_size, config, shared_data, tune_flag, dataset_list, gene_dict, partitions, val_num = args
    #print(rank)
    
    batch_list, seq_list, order_list, coord_list, label_list = shared_data

    partition = partitions[rank]
    model = OmicsFormer(**config).cuda(rank)
    if rank == 0:
        total_params = 0
        for param_tensor in model.state_dict():
            param = model.state_dict()[param_tensor]
            total_params += torch.numel(param)
        print(total_params)
    if config['ddp']:
        setup(rank, world_size)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # gmm_params = []
    # other_params = []
    # for pname, p in model.nondisc_parameters():
    #     if False:  # 'mean' in pname or 'std' in pname:
    #         print(pname)
    #         gmm_params += [p]
    #     else:
    #         other_params += [p]
    # optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    optim = torch.optim.AdamW(model.nondisc_parameters(), lr=config['lr'], weight_decay = config['wd'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.7, patience=5, verbose=True)

    train_loss = []
    valid_loss = []
    loss_curve = []
    
    for epoch in tqdm(range(config['epochs'])):
        start_time = time.time()
        epoch_loss = []
        model.train()

        if config['batch_mixing']:
            for i in range(len(batch_list)):
                idx = np.random.permutation(config['batch_num'])[:(config['max_batch_size'] // 200)]
                ratio = (config['max_batch_size']-len(idx)*10) / sum([len(batch_list[j]) for j in idx])
                temp_seq = []
                temp_coord = []
                temp_batch = []
                temp_dataset = []
                for j in idx:
                    cur = torch.randperm(batch_list[j].shape[0])[: 10+int(ratio * batch_list[j].shape[0])]
                    x = torch.sparse_csr_tensor(seq_list[0][j], seq_list[1][j], seq_list[2][j],
                                                seq_list[3][j].tolist()).to_sparse().float().coalesce()
                    temp_seq.append(x.index_select(0, cur))
                    temp_coord.append(coord_list[j][cur])
                    temp_batch.append(batch_list[j][cur])
                    temp_dataset.append(dataset_list[j][cur])

                input_dict = {'x_seq': torch.cat(temp_seq).cuda(rank),
                              'x_masked_seq': torch.cat(temp_seq).cuda(rank),
                              'coord': torch.cat(temp_coord).cuda(rank),
                               'batch': torch.cat(temp_batch).cuda(rank),
                              'dataset': torch.cat(temp_dataset).cuda(rank),
                              'gene_mask': torch.ones([x.shape[1]]).bool().cuda(rank),
                              'ecs': config['ecs']}
                x_dict = XDict(input_dict)
                out_dict, loss = model(x_dict)

                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss.append(out_dict['target_loss'])
                del loss, x_dict, x
 
        else:
            for i in random.sample(partition, len(partition)):#partition:
                #print(i)
                x = torch.sparse_csr_tensor(seq_list[0][i], seq_list[1][i], seq_list[2][i],
                                        seq_list[3][i].tolist()).to_sparse().float().coalesce()
     
                x_dict = XDict({'x_seq': x.cuda(rank),  # seq_list[i].cuda(rank),
                                'batch': batch_list[i].cuda(rank),
                                'coord': coord_list[i].cuda(rank),
                                'gene_mask': torch.ones([x.shape[1]]).bool().cuda(),
                                'dataset': dataset_list[i].cuda(rank),
                                #'dropout': np.random.beta(config['beta'][0], config['beta'][1]),
                                })
                if config['flip'] and random.random()>0.85:
                    del x_dict['batch']
                if config['flip'] and random.random()>0.85:
                    del x_dict['dataset']
                out_dict, loss = model(x_dict)
                if out_dict['projection'] is not None:
                    cl_loss = info_nce_loss(out_dict['projection'], temperature=config['temperature']).unsqueeze(-1).to('cuda')
                    loss = loss + config['w_cl'] * cl_loss.item()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optim.step()
                optim.zero_grad()
                epoch_loss.append(out_dict['target_loss'])
                del loss, x_dict, x
                
        train_loss.append(sum(epoch_loss) / len(epoch_loss))       

        if epoch>80:
            for param_group in optim.param_groups:
                param_group['lr'] = param_group['lr'] * 0.999
        elif epoch<5:
            for param_group in optim.param_groups:
                param_group['lr'] = config['lr'] * (epoch+1)/5


    if rank == 0:
        if not tune_flag:
            if False:#val_num>0:
                torch.save(best_model_weights, f'./results/{config["name"]}_wi.pt')
                model.load_state_dict(best_model_weights)
                pass
            else:
                if config['reg_token']:
                    torch.save(model.state_dict(), f'./results/{config["name"]}_with_token_test.pt')
                else:
                    torch.save(model.state_dict(), f'./results/{config["name"]}_without_token_test.pt')

        # Inference
        '''c = []

        with torch.no_grad():
            for i in range(len(batch_list) - val_num, len(batch_list)):#range(len(batch_list)):
                try:
                    x = torch.sparse_csr_tensor(seq_list[0][i], seq_list[1][i],
                                            seq_list[2][i],
                                            seq_list[3][i].tolist()).to_sparse().float().coalesce()
                    x_dict = XDict({'x_seq': x.cuda(),  # seq_list[i].cuda(rank),
                                    'batch': batch_list[i].cuda(),
                                    'coord': coord_list[i].cuda(),
                                    'gene_mask': torch.ones([x.shape[1]]).bool().cuda(),
                                     'dataset': dataset_list[i].cuda(rank),
                                    })
                    y = x_dict['x_seq'].to_dense()
                    out_dict, loss = model(x_dict)
                    c.append(CountCorr(out_dict['recon'], y).cpu().item())
                    # print(y, out_dict['recon'], corr(out_dict['recon'], y))
                except:
                    print(i, x.shape)

            del loss, out_dict
        torch.cuda.empty_cache()
        print('Validation Pearson:', sum(c) / len(c))'''

        c = []
        res = []
        model.eval()
        with torch.no_grad():
            for i in range(len(batch_list)):
                x = torch.sparse_csr_tensor(seq_list[0][i], seq_list[1][i],
                                        seq_list[2][i],
                                        seq_list[3][i].tolist()).to_sparse().float().coalesce()
                x_dict = XDict({'x_seq': x.cuda(),  # seq_list[i].cuda(rank),
                                'batch': batch_list[i].cuda(),
                                'coord': coord_list[i].cuda(),
                                'gene_mask': torch.ones([x.shape[1]]).bool().cuda(),
                                 'dataset': dataset_list[i].cuda(),
                                })
                y = x_dict['x_seq'].to_dense()
                #print(y.shape)
                #out_dict, loss = model(x_dict)
                rep, atten = model.get_representation(x_dict)
                #c.append(CountCorr(out_dict['recon'], y).cpu().item())
                res.append(rep['h'].cpu())
                #print(out_dict['recon']/)
                #c.append(CountCorr(out_dict['recon'], y).cpu().item())
                #res.append(out_dict['latent'].cpu())
            del model
        torch.cuda.empty_cache()

        res = torch.cat(res, dim=0).numpy()
        data_eval = ad.AnnData(X=res,
                               obs=pd.DataFrame({'batch': torch.cat(batch_list), 'order': torch.cat(order_list)}))  # data[order]
        data_eval.obs['cell_type'] = torch.cat(label_list).numpy().tolist()
        data_eval.obs['cell_type'] = data_eval.obs['cell_type'].astype('category')
        data_eval.obs['batch'] = data_eval.obs['batch'].astype('category')
        data_eval.obsm['X_cellbert'] = res
        if not tune_flag:
            data_eval.write_h5ad('./results/' + config["name"]  + '.h5ad', compression=hdf5plugin.FILTERS["zstd"])

        #df = minimum_eval(data_eval)
        #print(df)
        cres = clustering_eval(data_eval)
        print(cres)
        if tune_flag:
            wandb.log({'test_pearson': sum(c) / len(c)})
            wandb.log(cres)
            #wandb.log({'graph_conn': df.T['graph_conn'].values[0]})
            wandb.finish()
        data_eval.write_h5ad(config["name"] + str(time.time())[-3:]+'.h5ad', compression=hdf5plugin.FILTERS["zstd"])
        del res, y
        torch.cuda.empty_cache()
        if config['ddp']:
            cleanup()

def main(config=None):
    global tune_flag, add_dict
    global gene_list, batch_labels, seq_list, order_list, gene_dict, dataset_list, coord_list, \
        val_num, tune_flag, label_list, gene_list, batch_list
        
    seed_torch(42)
    # mp.set_start_method('spawn')
    # tune_flag = True if config is None else False
    if tune_flag:
        wandb.init(
            # set the wandb project where this run will be logged
            group="integration-search-reg_number",
        )
        config = wandb.config

    if add_dict is not None:
        config.update(add_dict)
    config["batch_num"] = batch_labels.max() + 1
    config['gene_list'] = gene_list
    config['dataset_num'] = ndataset
    if not tune_flag:
        os.makedirs('results', exist_ok=True)
        with open('./results/' + config['name']+'.config.pkl', 'wb') as f:
            pickle.dump(config, f)
    val_num = config['val_num']
    world_size = torch.cuda.device_count()
    if val_num > 0:
        partitions = balanced_partition(batch_list[:-val_num], world_size, config['max_batch_size'])
    else:
        partitions = balanced_partition(batch_list, 1, config['max_batch_size'])
    for i in range(len(batch_list)):
        batch_list[i] = batch_list[i].share_memory_()
        order_list[i] = order_list[i].share_memory_()
        coord_list[i] = coord_list[i].share_memory_()
        label_list[i] = label_list[i].share_memory_()
    for i in range(4):
        for j in range(len(seq_list[i])):
            seq_list[i][j] = seq_list[i][j].share_memory_()
    shared_data = [batch_list, seq_list, order_list, coord_list, label_list]

    if config['ddp']:
        mp.spawn(train, args=(world_size, config, shared_data, tune_flag, dataset_list, gene_dict, partitions, val_num),
             nprocs=world_size, join=True)
    else:
        partitions = balanced_partition(batch_list, 1)
        train(0, 1, config, shared_data, tune_flag, dataset_list, gene_dict, partitions, val_num)
    # mp.spawn(train, args=(world_size, config, partitions, shared_data), nprocs=world_size, join=True)



from scipy import sparse


mp.set_sharing_strategy('file_system')
mp.set_start_method('spawn', force=True)
parser = argparse.ArgumentParser()
parser.add_argument("--tune", action='store_true', default=False)
parser.add_argument("--reg_token", action='store_true', default=False)
args = parser.parse_args()

args.tune = False
#dataset_name = 'pancreas'#'CellBert_v1'#'HLCA_zstd'#'CellBert_subset' #
dataset_name = 'lung'


if dataset_name in ['CellBert_v0', 'CellBert_subset']: # datasets haven't been prerpocessed
    data = ad.read_h5ad(f'/home/ec2-user/Project/new_integration/{dataset_name}.h5ad')
    gene_list = data.var.index.to_list()
    gene_to_idx = {gene_list[i]: i for i in range(len(gene_list))}
    with open(f'/home/ec2-user/Project/new_integration/{dataset_name}.gene.json') as f:
        gene_dict = json.load(f)
        new_gene_dict = {}
        for k in gene_dict.keys():
            gene_mask = torch.zeros(len(gene_list)).int()
            gene_mask[[gene_to_idx[gene] for gene in gene_dict[k]]] = 1
            new_gene_dict[k] = gene_mask.bool()
        gene_dict = new_gene_dict
    data.obs['batch'] = data.obs['batch_label']
elif dataset_name in ['HLCA_zstd']:
    data = sc.read_h5ad('2000.h5ad')
    if 'platform' not in data.obs:
        data.obs['platform'] = 'scRNA-seq'
    if 'Dataset' not in data.obs:
        data.obs['Dataset'] = data.obs['study']
    if 'batch_label' in data.obs:
        data.obs['batch'] = data.obs['batch_label']
    # if 'cell_type' not in data.obs:
    #     data.obs['cell_type'] = 'NA'
    #sc.pp.highly_variable_genes(data, n_top_genes=4000, subset=True, flavor='seurat_v3')
    gene_list = data.var.index.to_list()
    gene_dict = dict(zip(data.obs['Dataset'].unique(), [torch.ones(len(gene_list)).bool()] * data.obs['Dataset'].nunique()))
    ndataset = data.obs['Dataset'].nunique()
elif dataset_name in ['pancreas']:
    data = sc.read_h5ad('pancreas.ha5d')
    data.obs['cell_type'] = data.obs['celltype']
    # if 'cell_type' not in data.obs:
    data.obs['batch'] = data.obs['tech']
    data.obs['Dataset'] = '1'
    data.obs['platform'] = 'scRNA-seq'
    gene_list = data.var.index.to_list()
    data.X = sparse.csr_matrix(data.X)
    gene_dict = dict(zip(data.obs['Dataset'].unique(), [torch.ones(len(gene_list)).bool()] * data.obs['Dataset'].nunique()))
    ndataset = data.obs['Dataset'].nunique()
elif dataset_name in ['immune']:
    data = sc.read_h5ad('immune.h5ad')
    data.obs['cell_type'] = data.obs['final_annotation']
    data.obs['Dataset'] = data.obs['study']
    data.obs['platform'] = 'scRNA-seq'
    gene_list = data.var.index.to_list()
    data.X = sparse.csr_matrix(data.X)
    gene_dict = dict(zip(data.obs['Dataset'].unique(), [torch.ones(len(gene_list)).bool()] * data.obs['Dataset'].nunique()))
    ndataset = data.obs['Dataset'].nunique()
elif dataset_name in ['lung']:
    data = sc.read_h5ad('lung.h5ad')
    data.obs['Dataset'] = data.obs['dataset']
    data.obs['platform'] = 'scRNA-seq'
    gene_list = data.var.index.to_list()
    data.X = sparse.csr_matrix(data.X)
    gene_dict = dict(zip(data.obs['Dataset'].unique(), [torch.ones(len(gene_list)).bool()] * data.obs['Dataset'].nunique()))
    ndataset = data.obs['Dataset'].nunique()
    
else: # prerpocessed datasets
    data = ad.read_h5ad(f'../data/{dataset_name}.h5ad')# , backed='r') # backed can run but too slow
    with open(f'../data/{dataset_name}.gene.pkl', 'rb') as f:
        gene_dict = pickle.load(f)
    gene_list = data.var.index.to_list()



seq_list, batch_list, batch_labels, order_list, dataset_list, coord_list, label_list = data_setup(data)

out_dim = len(gene_list)
del data

from datetime import datetime


if args.tune:
    tune_flag = True
    sweep_configuration = {
        'method': 'grid',
        'name': 'integration_31',
        'metric': {
            'goal': 'maximize',
            'name': 'ari'
        },
        'parameters': {
            "enc_hid": {'values': [1024]},
            "mask_feature_rate": {'values': [0.25, 0.3, 0.5, 0.7]},
            "latent_mod": {'values': ['vae']},
            "flip": {'values': [False]},
            "w_kl": {'values': [.1, .2 ,.3, .4, .5]},
            "reg_token_number": {'values': [2, 4, 8, 16]},
            "post_hoc" : {'values': [True]}
        }
    }
    add_dict = {
        'flip': False,
        'dae': True,
        'beta': (1., 1.),
        'mask_feature_rate': 0.3,
        "post_latent_dim": 512,
        "enc_layers": 2,
        "lamda": 0.15,
        "enc_hid": 1024,
        "dec_hid": 1024,
        "latent_mod": 'vae',
        "mask_node_rate": 0.8,
        "norm": "layernorm",
        "dec_layers": 2,
        "dsbn": False,
        "batch_mixing": False,
        "ecs": False,
        "enc_mod": "transformer",
        "dec_mod": 'nbmlp',
        "num_clusters": 16,
        "model_dropout": 0.2,
        "dataset": "HLCA",
        "architecture": "OmicsFormer",
        "epochs": 250,
        "drop_node_rate": 0.3,
        "w_li": 1.,
        "w_en": 1.,
        "lr": 2e-4,
        "wd": 1e-8,
        "out_dim": out_dim,
        "ddp": False,
        'cat_pe': False,
        'val_num': 2,
        'dar': False,
        'max_batch_size': 10000,
        'input_covariate': False,
        'w_kl':.4,
        'post_hoc':True,
        'reg_token': args.reg_token
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='CellBertv6')
    wandb.agent(sweep_id=sweep_id, function=main, count=200)
else:
    tune_flag = False  
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    config = {
        "name": "pan-{}".format(current_time),
        "enc_hid": 1024,
        "enc_layers": 2,
        "latent_mod": 'vae',
        "post_latent_dim": 1024,
        "dec_hid": 512,
        "mask_node_rate": 0.8,
        "mask_feature_rate": 0.3,
        "lamda": 0.15,
        "beta": (1., 1.),
        'dae': True,
        'flip': False,
        "norm": "layernorm",
        "dec_layers": 2,
        "dsbn": False,
        "batch_mixing": False,
        "ecs": False,
        "enc_mod": "transformer",
        "dec_mod": 'nbmlp',
        "num_clusters": 16,
        "model_dropout": 0.2,
        "dataset": "pancreas",
        "architecture": "OmicsFormer",
        "epochs": 250,
        "drop_node_rate": 0.3,
        "w_li": 1.,
        "w_en": 1.,
        "lr": 2e-4,
        "wd": 1e-8,
        "w_kl":0.4,
        'post_hoc':True,
        "out_dim": out_dim,
        "ddp": False,
        'cat_pe': False,
        'val_num': 0,
        'dar': False,
        'max_batch_size': 10000,
        'input_covariate': False,
        'reg_token': args.reg_token,
        'reg_token_number': 8,
        'projection': False,
        'hidden_drop': 0.2,
        'w_cl': 0.5,
        'temperature': 0.07
    }
    add_dict = None
    main(config)