import scanpy as sc
import scib
import numpy as np
import torch
import torch.nn.functional as F
# import rapids_singlecell as rsc
from scipy.sparse import csr_matrix
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall 
from scib.metrics.ari import ari
from scib.metrics.nmi import nmi
# rmm.reinitialize(
#     managed_memory=True, # Allows oversubscription
#     devices=2, # GPU device IDs to register. By default registers only GPU 0.
# )
# cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def downstream_eval(task, pred_labels, true_labels, num_classes=None, eval_mask=None, dim=1, 
                    normalize=True, top_de_dict=None, batch_labels=None, control_level=None,
                    topk=20, **kwargs):
    if task == 'annotation':
        return annotation_eval(pred_labels, true_labels, num_classes)
    elif task == 'denoising':
        return denoising_eval(pred_labels, true_labels, eval_mask, normalize)
    elif task == 'imputation':
        return imputation_eval(pred_labels, true_labels, dim)
    elif task == 'perturbation_prediction':
        return perturbation_prediction_eval(pred_labels, true_labels, top_de_dict, batch_labels, 
                                            control_level, topk)
    else:
        raise NotImplementedError(f"{task} should be chosen from ['annotation', 'denoising', 'imputation', 'perturbation_prediction']")

def CountCorr(y_true, y_pred):
    y_true = torch.log1p(y_true)
    y_pred = torch.log1p(y_pred)
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) / torch.sqrt(
        torch.sum(y_pred_c * y_pred_c, 1)))
    return pearson

def PearsonCorr(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) 
                         / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1)))
    return pearson

def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c) / torch.sqrt(torch.sum(y_true_c * y_true_c)) 
                         / torch.sqrt(torch.sum(y_pred_c * y_pred_c)))
    return pearson

def clustering_eval(adata, cluster_key='leiden', label_key='cell_type'):
    import rapids_singlecell as rsc
    print('Start building knn.')
    sc.pp.neighbors(adata, use_rep='X_cellbert', method='rapids', random_state=42)
    best_ari = -1
    best_nmi = -1
    for res in range(1, 15, 1):
        res = res / 10
        rsc.tl.leiden(adata, resolution=res, key_added=cluster_key)
        ari_score = ari(adata, cluster_key=cluster_key, label_key=label_key)
        if ari_score > best_ari:
            best_ari = ari_score
        nmi_score = nmi(adata, cluster_key=cluster_key, label_key=label_key)
        if nmi_score > best_nmi:
            best_nmi = nmi_score
    return {'ari': best_ari, 'nmi':best_nmi}

def minimum_eval(adata):
    print('Start building knn.')
    sc.pp.neighbors(adata, use_rep='X_cellbert', method='rapids')
    return scib.metrics.metrics(adata, adata, "batch", "ann_level_3", embed='X_cellbert', cluster_key="cluster",
                         #organism='human', ari_=True, nmi_=True, pcr_=True, graph_conn_=True)
    organism = 'human', graph_conn_ = True)

def annotation_eval(pred_labels, true_labels, num_classes=None):
    num_classes = len(true_labels.unique()) if num_classes is None else num_classes
    acc = multiclass_accuracy(pred_labels, true_labels, num_classes).cpu().item()
    f1_score = multiclass_f1_score(pred_labels, true_labels, num_classes).cpu().item()
    precision = multiclass_precision(pred_labels, true_labels, num_classes).cpu().item()
    recall = multiclass_recall(pred_labels, true_labels, num_classes).cpu().item()
    return {'acc': acc, 'f1_score': f1_score, 'precision': precision, 'recall': recall}

def normalize_counts(counts):
    counts = F.relu(counts / counts.sum(1, keepdim=True) * 1e4)
    return torch.log1p(counts)

def denoising_eval(pred_labels, true_labels, eval_mask=None, normalize=True):
    if normalize:
        true_labels = normalize_counts(true_labels)
        pred_labels = normalize_counts(pred_labels)
    if eval_mask is not None:
        true_labels = true_labels[eval_mask]
        pred_labels = pred_labels[eval_mask]
        nz_idx = torch.nonzero(true_labels, as_tuple=True)
        true_labels = true_labels#[nz_idx]
        pred_labels = pred_labels#[nz_idx]
        corr = PearsonCorr1d(pred_labels, true_labels).item()
        cos = F.cosine_similarity(pred_labels, true_labels, dim=0).item()
    else:
        corr = PearsonCorr(pred_labels, true_labels).item()
        cos = F.cosine_similarity(pred_labels, true_labels, dim=1).mean().item()
    mse = F.mse_loss(pred_labels, true_labels).item()
    rmse = np.sqrt(mse)
    mae = F.l1_loss(pred_labels, true_labels).item()
    return {'mse': mse, 'rmse':rmse, 'mae':mae, 'corr':corr, 'cos': cos}

def imputation_eval(pred_labels, true_labels, dim=1):
    mse = []
    rmse = []
    rmsle = []
    mae = []
    corr = []
    cos = []
#     if ((true_labels - true_labels.int().float())<1e-6).all():
#         true_labels = torch.log1p(true_labels)
#         pred_labels = torch.log1p(pred_labels)
    for i in range(true_labels.shape[dim]):
        true_vec = true_labels[i] if dim == 0 else true_labels[:,i]
        pred_vec = pred_labels[i] if dim == 0 else F.relu(pred_labels[:,i])
        nz_idx, _ = torch.nonzero(true_labels, as_tuple=True)
        true_nz = true_vec[nz_idx]
        pred_nz = pred_vec[nz_idx]
        mse.append(F.mse_loss(pred_nz, true_nz).item())
        rmse.append(np.sqrt(mse))
        rmsle.append(np.sqrt(F.mse_loss(torch.log(pred_nz+1), torch.log(true_nz+1)).item()))
        mae.append(F.l1_loss(pred_nz, true_nz).item())
        corr.append(PearsonCorr1d(pred_nz, true_nz).item())
        cos.append(F.cosine_similarity(pred_nz, true_nz, dim=0).item())
    rmse = np.concatenate(rmse)
    return {
        'mse': sum(mse)/len(mse), 
        'rmse': sum(rmse)/len(rmse), 
        'rmsle': sum(rmsle)/len(rmsle), 
        'mae': sum(mae)/len(mae), 
        'corr': sum(corr)/len(corr),
        'cos': sum(cos)/len(cos),
    }

def perturbation_prediction_eval(pred_labels, true_labels, top_de_dict=None, batch_labels=None, 
                                 control_level=None, topk=20):
    true_labels = true_labels.cuda()
    pred_labels = pred_labels.cuda()
    if control_level is not None:
        control_level = control_level.cuda()
        import ipdb
        ipdb.set_trace()
        if len(control_level.shape) == 1:
            control_level = control_level.unsqueeze(0)
        true_labels = true_labels - control_level
        pred_labels = pred_labels - control_level
    
    if batch_labels is not None:
        all_rmse = []
        all_corr = []
        all_cos = []
        for batch in batch_labels:
            batch_truth = true_labels[batch_labels == batch].mean(0)
            batch_pred = pred_labels[batch_labels == batch].mean(0)
            all_rmse.append(np.sqrt(F.mse_loss(batch_pred, batch_truth).item()))
            all_corr.append(PearsonCorr1d(batch_pred, batch_truth).item())
            all_cos.append(torch.mean(F.cosine_similarity(batch_pred, batch_truth, dim=0)).item())
        all_rmse = sum(all_rmse) / len(all_rmse)
        all_corr = sum(all_corr) / len(all_corr)
        all_cos = sum(all_cos) / len(all_cos)

    else:        
        all_rmse = np.sqrt(F.mse_loss(pred_labels, true_labels).item())
        all_corr = PearsonCorr(pred_labels, true_labels).item()
        all_cos = torch.mean(F.cosine_similarity(pred_labels, true_labels, dim=1)).item()
    
    if top_de_dict is not None:
        rmse = []
        corr = []
        cos = []
        for k, v in top_de_dict.items():
            assert len(v) >= topk, f"Expect topk be <= {len(v)}, but found {topk}"
            true_degs = true_labels[:, v[:topk]]
            pred_degs = pred_labels[:, v[:topk]]
            if batch_labels is not None:
                batch_rmse = []
                batch_corr = []
                batch_cos = []
                for batch in batch_labels:
                    batch_truth = true_degs[batch_labels == batch].mean(0)
                    batch_pred = pred_degs[batch_labels == batch].mean(0)
                    batch_rmse.append(np.sqrt(F.mse_loss(batch_pred, batch_truth).item()))
                    batch_corr.append(PearsonCorr1d(batch_pred, batch_truth).item())
                    batch_cos.append(torch.mean(F.cosine_similarity(batch_pred, batch_truth, dim=0)).item())
                rmse.append(sum(batch_rmse) / len(batch_rmse))
                corr.append(sum(batch_corr) / len(batch_corr))
                cos.append(sum(batch_cos) / len(batch_cos))
            else:
                rmse.append(np.sqrt(F.mse_loss(pred_degs, true_degs).item()))
                cos.append(torch.mean(F.cosine_similarity(pred_degs, true_degs, dim=1)).item())
                if true_degs.sum(1).min() != 0 and pred_degs.sum(1).min() != 0:
                    corr.append(PearsonCorr(pred_degs, true_degs).item())
        if len(corr) == 0:
            corr = [0]
    else:
        rmse = corr = cos = [0]
    return {
        'all_rmse': all_rmse,
        'all_corr': all_corr,
        'all_cos': all_cos,
        'top_de_rmse': sum(rmse)/len(rmse), 
        'top_de_corr': sum(corr)/len(corr),
        'top_de_cos': sum(cos)/len(cos),
    }
