import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..embedder import OmicsEmbeddingLayer
from ..utils.mask import MaskBuilder, NullMaskBuilder, HiddenMaskBuilder
from ..encoder import setup_encoder
from ..decoder import setup_decoder
from ..latent import LatentModel
from ..objective import Objectives
from ..head import setup_head

class OmicsFormer(nn.Module):
    def __init__(self, gene_list, enc_mod, enc_hid, enc_layers, post_latent_dim, dec_mod, dec_hid, dec_layers, batch_num,
                 out_dim, mask_type='input', model_dropout=0.3, activation='gelu', norm='layernorm', enc_head=8, mask_node_rate=0.5,
                 mask_feature_rate=0.8, drop_node_rate=0., cat_dim=None, conti_dim=None, pe_type='sin', cat_pe=True,
                 gene_emb=None, latent_mod='vae', w_li=1., w_en=1., w_ce=1., input_drop_type=None, input_drop_rate=0.1,
                 seed=10, head_type=None, **kwargs):
        super(OmicsFormer, self).__init__()
        self.embedder = OmicsEmbeddingLayer(gene_list, enc_hid, norm, activation, model_dropout,
                                            pe_type, cat_pe, gene_emb)
        self.mask_type = mask_type
        if mask_node_rate > 0 and mask_feature_rate > 0:
            if mask_type == 'input':
                self.mask_model = MaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate)
            elif mask_type == 'hidden':
                self.mask_model = HiddenMaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate)
            else:
                raise NotImplementedError(f"Only support mask_type in ['input', 'hidden'], but got {mask_type}")
        else:
            self.mask_model = NullMaskBuilder(drop_node_rate)
        self.encoder = setup_encoder(enc_mod, enc_hid, enc_layers, model_dropout, activation, norm, enc_head)

        self.latent = LatentModel()
        if latent_mod=='vae':
            self.latent.add_layer(type='vae', enc_hid=enc_hid, latent_dim=post_latent_dim)
        elif latent_mod=='ae':
            self.latent.add_layer(type='merge', conti_dim=enc_hid, cat_dim=0, post_latent_dim=post_latent_dim)
        elif latent_mod=='gmvae':
            self.latent.add_layer(type='gmvae', enc_hid=enc_hid, latent_dim=post_latent_dim, batch_num=batch_num,
                                  w_li=w_li, w_en=w_en, w_ce=w_ce, dropout=model_dropout, num_layers=dec_layers,
                                  gumbel_softmax=kwargs['gumbel_softmax'], num_clusters=kwargs['num_clusters'])
        elif latent_mod=='vqvae':
            self.latent.add_layer(type='vqvae', enc_hid=enc_hid, latent_dim=post_latent_dim, ema_flag = kwargs['ema_flag'],
                                  num_categories=kwargs['num_categories'], w_commit=kwargs['w_commit'], w_vq=kwargs['w_vq'],
                                  decay=kwargs['decay'])
        elif latent_mod=='split':
            self.latent.add_layer(type='split', enc_hid=enc_hid, latent_dim=None, conti_dim=conti_dim, cat_dim=cat_dim)
            self.latent.add_layer(type='merge', conti_dim=conti_dim, cat_dim=cat_dim, post_latent_dim=post_latent_dim)
        self.head_type = head_type
        if head_type is not None:
            self.head = setup_head(head_type, post_latent_dim, dec_hid, out_dim, dec_layers,
                                   model_dropout, norm, batch_num)
        else:
            self.decoder = setup_decoder(dec_mod, post_latent_dim, dec_hid, out_dim, dec_layers,
                                         model_dropout, norm, batch_num)
            self.objective = Objectives([{'type': 'recon'}])

    def forward(self, x_dict, input_gene_list=None):
        if self.mask_type == 'input':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.embedder(x_dict, input_gene_list)
        #x_dict['h'] = self.tempemb(x_dict['x_seq'].to_dense())
        if self.mask_type == 'hidden':
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict['h'] = self.encoder(x_dict)['hidden']
        x_dict['h'], latent_loss = self.latent(x_dict)
        if self.head_type is not None:
            out_dict, loss = self.head(x_dict)
        else:
            out_dict = self.decoder(x_dict)
            loss = latent_loss + self.objective(out_dict, x_dict)
        return out_dict, loss
