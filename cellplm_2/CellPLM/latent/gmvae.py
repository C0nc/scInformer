import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layer.flowformer import FlowformerLayer
import torch.nn.init as init

### Reference: https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Networks.py ###

class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


# Sample from a Gaussian distribution
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z

class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            GumbelSoftmax(x_dim, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, 512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    def forward(self, x, temperature=1.0, hard=0):
        # x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        y_mu, y_var = self.pzy(y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y,
                  'y_mean': y_mu, 'y_var': y_var}
        return output

class GMVAELatentLayer(nn.Module):
    def __init__(self, enc_hid, latent_dim, num_clusters, hard=False,
                 w_li=1., w_en=1., lamda=0.5, **kwargs):
        super(GMVAELatentLayer, self).__init__()

        self.hard = hard
        self.inference = InferenceNet(enc_hid, latent_dim, num_clusters)
        self.w_li = w_li
        self.w_en = w_en
        self.lamda = lamda
        self.eps = 1e-8
        self.num_clusters = num_clusters
        self.is_adversarial = False

    def forward(self, x_dict, temperature=1.0):
        if self.training:
            out_dict = self.inference(x_dict['h'], temperature, self.hard)
            z = out_dict['gaussian']
            loss = self.unlabeled_loss(out_dict)
            return z, loss
        else:
            out_dict = self.inference(x_dict['h'], temperature, True)
            z = out_dict['mean']
            return z, self.unlabeled_loss(out_dict)

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
           loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
           z: (array) array containing the gaussian latent variable
           z_mu: (array) array containing the mean of the inference model
           z_var: (array) array containing the variance of the inference model
           z_mu_prior: (array) array containing the prior mean of the generative model
           z_var_prior: (array) array containing the prior variance of the generative mode

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def unlabeled_loss(self, out_net):
        """Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z = out_net['gaussian']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # gaussian loss
        # print('gaussian', self.gaussian_loss(z, mu, var, y_mu, y_var))
        loss_gauss = max(self.lamda, self.gaussian_loss(z, mu, var, y_mu, y_var))

        # print('cat', -self.entropy(logits, prob_cat) - np.log(1/self.num_clusters))
        # categorical loss
        loss_cat = max(self.lamda, -self.entropy(logits, prob_cat) - np.log(1/self.num_clusters))

        # total loss
        loss_total = self.w_li * loss_gauss + self.w_en * loss_cat

        return loss_total