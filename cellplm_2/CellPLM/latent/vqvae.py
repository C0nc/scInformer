import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

def compute_distance(inputs_flatten, codebook):
    codebook_sqr = torch.sum(codebook ** 2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
    distances = torch.addmm(codebook_sqr + inputs_sqr,
        inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
    return distances

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)
            distances = compute_distance(inputs_flatten, codebook)
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)
        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)
        return (grad_inputs, grad_codebook)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e):
        latents = vq(z_e, self.embedding.weight)
        return latents

    def straight_through(self, z_e):
        z_q, indices = vq_st(z_e, self.embedding.weight.detach())
        z_q_bar = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_bar = z_q_bar.view_as(z_e)
        return z_q, z_q_bar

class ExponentialMovingAverage(nn.Module):
    def __init__(self, init_value, decay):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average

class VQVAELatentLayer(nn.Module):
    def __init__(self, enc_hid, latent_dim, num_categories=64, w_commit=1., ema_flag=False, w_vq=1., decay=0.99, **kwargs):
        super().__init__()
        self.fc_layer = nn.Linear(enc_hid, latent_dim)
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.w_commit = w_commit
        self.w_vq = w_vq
        self.eps = 1e-10
        self.ema_flag = ema_flag
        if ema_flag:
            codebook = torch.empty(num_categories, latent_dim)
            nn.init.xavier_uniform_(codebook)
            self.register_buffer("codebook", codebook)
            self.ema_dw = ExponentialMovingAverage(self.codebook, decay)
            self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((num_categories,)), decay)
        else:
            self.codebook = VQEmbedding(num_categories, latent_dim)
        
    def quantize(self, encoding_indices):
        return F.embedding(encoding_indices, self.codebook)

    def ema_forward(self, z_e):
        flat_z_e = z_e.reshape(-1, self.latent_dim)
        distances = compute_distance(flat_z_e, self.codebook)
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_bar = self.quantize(encoding_indices).view_as(z_e)
        # straight through
        z_q = z_e + (z_q_bar - z_e).detach()

        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_categories).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.eps) /
                                      (n + self.num_categories * self.eps) * n)
            dw = torch.matmul(encodings.t(), flat_z_e) # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.codebook.data = normalised_updated_ema_w
        return z_q, z_q_bar
        
    def forward(self, x_dict):
        h = self.fc_layer(x_dict['h'])
        if self.ema_flag:
            z_q, z_q_bar = self.ema_forward(h)
            loss_vq = 0
        else:
            z_q, z_q_bar = self.codebook.straight_through(h)
            # vector quantization loss
            loss_vq = F.mse_loss(z_q_bar, h.detach())

        # commitment loss
        loss_commit = F.mse_loss(h, z_q_bar.detach())
        loss = self.w_commit * loss_commit + self.w_vq * loss_vq
        return z_q, loss


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]