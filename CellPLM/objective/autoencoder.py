import torch
from torch import nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, lib_size=None, log_norm=False, **kwargs):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.lib_size = lib_size
        self.log_norm = log_norm
        self.downstream = None

    def forward(self, out_dict, x_dict):
        # if self.downstream == 'imputation':
        #     pred = out_dict['recon'][:, x_dict['gene_mask']]
        #     y = x_dict['label'][:, x_dict['y_gene_mask']]
        #     out_dict['pred'] = pred
        #     return self.reconstruction_loss(torch.softmax(pred, 1), y/(y.sum(1, keepdim=True)+1e-8))

        y = x_dict['x_seq'].to_dense()
        if self.lib_size is not None:
            y = y/y.sum(1)[:, None] * self.lib_size
        if self.log_norm:
            y = torch.log(y+1)
        # if 'gene_mask' in x_dict:
        #     return self.reconstruction_loss(out_dict['recon'][x_dict['input_mask'], x_dict['gene_mask']], y[:, x_dict['gene_mask']])
        # else:
        #     return self.reconstruction_loss(out_dict['recon'][x_dict['input_mask']], y)

        size_factor = y.sum(1, keepdim=True)
        pred = (size_factor * out_dict['recon'] * x_dict['input_mask'])[:, x_dict['gene_mask']]
        truth = (y * x_dict['input_mask'])[:, x_dict['gene_mask']]
        pred = pred[x_dict['input_mask'].sum(1)>0]
        truth = truth[x_dict['input_mask'].sum(1)>0]
        out_dict['pred'] = pred
        # pred = F.normalize(pred, p=2)
        # truth = F.normalize(truth, p=2)
        # return -(pred * truth).sum()
        # return self.reconstruction_loss(pred/(pred.sum(1, keepdim=True)+1e-8), truth/(truth.sum(1, keepdim=True)+1e-8))
        return self.reconstruction_loss(pred, truth)
        # return self.reconstruction_loss((out_dict['recon'] * x_dict['input_mask'])[:, x_dict['gene_mask']],
        #                                  (y * x_dict['input_mask'])[:, x_dict['gene_mask']])

