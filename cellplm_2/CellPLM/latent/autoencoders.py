import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import mixture
# from kmeans_pytorch import kmeans
from ..decoder import MLPDecoder

class SplitLatentLayer(nn.Module):
    def __init__(self, enc_hid, latent_dim=None, conti_dim=None, cat_dim=None, cont_l2_reg=0.01, cont_l1_reg=0.01, **kwargs):
        super().__init__()
        if conti_dim is None and cat_dim is None:
            assert latent_dim is not None, 'Latent dimension not specified!'
            self.hid_2lat = nn.Sequential(
                                nn.Linear(enc_hid, latent_dim),
                                nn.GELU(),
            )
        else:
            if conti_dim is not None and cat_dim is not None:
                if latent_dim is None and conti_dim + cat_dim != latent_dim:
                    logging.warning("latent_dim is ignored, since conti_dim and cat_dim are given.")
            elif cat_dim is None:
                conti_dim = latent_dim - cat_dim
            else:
                cat_dim = latent_dim - conti_dim

            latent_dim = None
            self.hid_2cont = nn.Sequential(
                                nn.Linear(enc_hid, conti_dim),
                                nn.GELU(),
            )
            self.hid_2cat = nn.Sequential(
                                nn.Linear(enc_hid, cat_dim),
                                nn.Softmax(1),
            )

        self.latent_dim = latent_dim
        self.conti_dim = conti_dim
        self.cat_dim = cat_dim
        self.is_adversarial = False
        self.cont_l1_reg = cont_l1_reg
        self.cont_l2_reg = cont_l2_reg

    def forward(self, x_dict=None):
        h = x_dict['h']
        if self.latent_dim is not None:
            h = self.hid_2lat(h)
            loss = 0
        else:
            h = torch.cat([self.hid_2cont(h), self.hid_2cat(h)], 1)
            params = torch.cat([x.view(-1) for x in self.hid_2cont.parameters()])
            loss = self.cont_l1_reg * torch.norm(params, 1) + self.cont_l2_reg * torch.norm(params, 2)
        return h, loss

class MergeLatentLayer(nn.Module):
    """
    Merge discrete and continuous dimensions to a new continious latent space
    """
    def __init__(self, conti_dim, cat_dim, post_latent_dim, **kwargs):
        super().__init__()

        self.lat_2lat = nn.Sequential(
                            nn.Linear(conti_dim + cat_dim, post_latent_dim),
                            # nn.ReLU(),
        )
        self.post_latent_dim = post_latent_dim
        self.conti_dim = conti_dim
        self.cat_dim = cat_dim
        self.is_adversarial = False

    def forward(self, x_dict):
        h = x_dict['h']
        return self.lat_2lat(h), 0

class VAELatentLayer(nn.Module):
    def __init__(self, enc_hid, latent_dim, kl_weight=1., warmup_step=10000, lamda=1.0, **kwargs):#400*160
        super().__init__()
        self.hid_2mu = nn.Linear(enc_hid, latent_dim)#, bias=False)
        self.hid_2sigma = nn.Linear(enc_hid, latent_dim)#, bias=False)
        self.kl_weight = 0#kl_weight
        self.max_kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step
        self.is_adversarial = False
        self.lamda = lamda

    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = self.kl_weight + self.max_kl_weight / self.warmup_step
        elif self.step_count == self.warmup_step:
            pass

    def forward(self, x_dict, var_eps=True):
        h = x_dict['h']
        mu = self.hid_2mu(h)
        log_var = torch.clamp(self.hid_2sigma(h), -5, 5) #+ 1e-4
        if var_eps:
            sigma = (torch.exp(log_var) + 1e-4).sqrt()
            log_var = 2 * torch.log(sigma)
        else:
            sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)

        if self.training:
            z = mu + sigma * eps
            kl_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean() * self.kl_weight
            if kl_loss < self.lamda:
                kl_loss = 0
            self.kl_schedule_step()
        else:
            z = mu
            kl_loss = 0
        return z, kl_loss
    
class GMM(nn.Module):
    def __init__(self, latent_dim, num_clusters):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros([1, latent_dim, num_clusters], dtype=torch.float32))
        self.std = nn.Parameter(torch.ones([1, latent_dim, num_clusters], dtype=torch.float32))
        self.N = latent_dim
        self.K = num_clusters
        self.eps = 1e-10

    def get_para(self):
        return self.mean, self.std
    
    def compute_log_likelihood(self, z_mean, z_logvar, cond_prob):
        term1 = torch.sum(-torch.log((self.std**2)*2*torch.pi), dim=1)*0.5
        term2 = torch.sum(-torch.div(torch.pow(z_mean.view(-1, self.N, 1) - self.mean, 2)
                                     +torch.exp(z_logvar).view(-1, self.N, 1), self.std**2), dim=1)*0.5
        prob = term2 + term1
        log_p = torch.mean(torch.mul(prob, cond_prob))
        return log_p
    
    def compute_prior(self, z):
        prob = torch.exp(torch.sum(-torch.log((self.std**2)*2*torch.pi)
                                   -torch.div(torch.pow(z.view(-1, self.N, 1) - self.mean, 2), self.std**2), dim=1)*0.5)
        pc = torch.div(prob, (torch.sum(prob, dim=-1)).view(-1,1) + self.eps)		
        return F.softmax(pc, dim=-1)
    
    def compute_entropy(self, cond_prob):
        return torch.mean(-torch.mul(cond_prob, torch.log(cond_prob + self.eps)))

    def compute_cross_entropy(self, cond_prob, pc):
        return torch.mean(-torch.mul(cond_prob, torch.log(pc + self.eps)))
    
    def forward(self):
        pass

class GMVAELatentLayer(GMM):
    def __init__(self, enc_hid, latent_dim, batch_num, num_layers=2, num_clusters=10, gumbel_softmax=False,
                 dropout=0., w_li=1., w_en=1., w_ce=1., **kwargs):
        super().__init__(latent_dim, num_clusters)
        self.hid_2mu = nn.Linear(enc_hid, latent_dim)
        self.hid_2sigma = nn.Linear(enc_hid, latent_dim)
        # self.classifier = MLPDecoder(enc_hid, latent_dim, num_clusters, num_layers, dropout, "layernorm", batch_num)
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_clusters)
        )

        self.num_clusters = num_clusters
        self.gumbel_softmax_flag = gumbel_softmax
        self.w_li = w_li
        self.w_en = w_en
        self.w_ce = w_ce
    
    def _init_params(self, z):
        # will be removed in later version
        device = z.device
        cluster_idx, mean = kmeans(X=z, num_clusters=self.num_clusters, device=device)
        mean = mean.to(device)
        var = []
        for i in range(self.num_clusters):
            var.append(torch.sum((z[cluster_idx == i] - mean[i])**2, dim=0, keepdim=True) 
                       / (sum(cluster_idx == i) - 1))
        var = torch.cat(var, dim=0)
        self.mean.data = mean.T.unsqueeze(0)
        self.std.data = torch.sqrt(var.T).unsqueeze(0)

    def gumbel_softmax(self, logits, temperature=1, eps=1e-10):
        y = logits - torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x_dict):
        # sample z
        h = x_dict['h']
        mu = self.hid_2mu(h)
        log_var = self.hid_2sigma(h)
        sigma = torch.exp(0.5 * log_var)
        noise = torch.randn_like(sigma)
        z = mu + sigma * noise

        # compute p(c|z)
        logits = self.classifier(x_dict['h'])

        if self.gumbel_softmax_flag:
            pc_z = self.gumbel_softmax(logits)
        else:
            pc_z = F.softmax(logits, dim=-1)
        self.pc_z = pc_z

        # compute loss
        if self.training:
            with torch.no_grad():
                pc = self.compute_prior(z)
            log_likelihood = self.compute_log_likelihood(mu, log_var, pc_z)
            entropy = self.compute_entropy(pc_z)
            cross_entropy = self.compute_cross_entropy(pc_z, pc)
            loss = self.w_li * log_likelihood + self.w_en * entropy + self.w_ce * cross_entropy
            if log_likelihood < -1e4 or torch.isnan(loss).any():
                import ipdb
                ipdb.set_trace()
        else:
            loss = 0
        return z, loss