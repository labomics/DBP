from os import path
from os.path import join as pj
import math
from re import S
import numpy as np
import torch as th
import torch.nn as nn

import functions.models as F
from torch.distributions.beta import Beta
from  torch.distributions.kl import kl_divergence
import modules.utils as utils


class Net(nn.Module):
    def __init__(self, o):
        super(Net, self).__init__()
        self.o = o
        self.sct = SCT(o)
        self.loss_calculator = LossCalculator(o)

    def forward(self, inputs):
        # *outputs, = self.sct(inputs)
        # loss = self.loss_calculator(inputs, *outputs)
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, cw_all, A, alpha, beta_a, beta_b, post_Ber, prior_Ber, w, A_mu, A_logvar, EW, EA = self.sct(inputs)
        loss = self.loss_calculator(inputs, x_r_pre, s_r_pre, z_mu, z_logvar, b, z_uni, A, alpha, beta_a, beta_b, post_Ber, prior_Ber, w, A_mu, A_logvar, EW, EA)
        return loss, cw_all


class SCT(nn.Module):
    def __init__(self, o):
        super(SCT, self).__init__()
        self.o = o
        self.sampling = False
        self.batch_correction = False
        self.b_centroid = None
        
        self.dim_z = o.dim_z
        self.dim_c = o.dim_c
        for m in o.mods:
            self.features = o.dims_h[m]
      
     
        self.alpha = 15 # Change for different datasets
          
        self.beta_a = nn.Parameter(th.ones(1, self.dim_c)*self.alpha)  
        self.beta_b = nn.Parameter(th.ones(1, self.dim_c))

        self.A_mu =  nn.Parameter(th.zeros(self.features, self.dim_c))
        self.A_var = 1e-8
        self.E1 =  nn.Parameter(th.zeros(self.dim_z-self.dim_c, self.features))
        self.E2 =  nn.Parameter(th.zeros(self.dim_z-self.dim_c, self.features))

        # Modality encoders q(z|x^m)
        x_encs = {}
        x_shared_enc = MLP(o.dims_enc_x+[o.dim_z*2+o.dim_c], hid_norm=o.norm, hid_drop=o.drop)
        
        for m in o.mods:
            x_indiv_enc = MLP([o.dims_h[m], o.dims_enc_x[0]], out_trans='mish', norm=o.norm,
                              drop=o.drop)
            x_encs[m] = nn.Sequential(x_indiv_enc, x_shared_enc)
        self.x_encs = nn.ModuleDict(x_encs)
        # # Modality decoder p(x^m|c, b)
        # self.x_dec = MLP([o.dim_z]+o.dims_dec_x+[sum(o.dims_h.values())], hid_norm=o.norm,
        #                  hid_drop=o.drop)

        # Subject encoder q(z|s)
        self.s_enc = MLP([o.dim_s]+o.dims_enc_s+[o.dim_z*2+o.dim_c], hid_norm=o.norm, hid_drop=o.drop)
    
        # Subject decoder p(s|b)
        # self.s_dec = MLP([o.dim_b]+o.dims_dec_s+[o.dim_s+sum(o.dims_h.values())*2], hid_norm=o.norm, hid_drop=o.drop)
        self.s_dec = MLP([o.dim_b]+o.dims_dec_s+[o.dim_s], hid_norm=o.norm, hid_drop=o.drop)

        # Chromosome encoders and decoders
        if "atac" in o.mods:
            chr_encs, chr_decs = [], []
            for dim_chr in o.dims_chr:
                chr_encs.append(MLP([dim_chr]+o.dims_enc_chr, hid_norm=o.norm, hid_drop=o.drop))
                chr_decs.append(MLP(o.dims_dec_chr+[dim_chr], hid_norm=o.norm, hid_drop=o.drop))
            self.chr_encs = nn.ModuleList(chr_encs)
            self.chr_decs = nn.ModuleList(chr_decs)
            self.chr_enc_cat_layer = Layer1D(o.dims_h["atac"], o.norm, "mish", o.drop)
            self.chr_dec_split_layer = Layer1D(o.dims_h["atac"], o.norm, "mish", o.drop)


    def forward(self, inputs):
        o = self.o
        x = inputs["x"]
        e = inputs["e"]
        s = None
        eps = 1e-10
        alpha = self.alpha
        beta_a = self.beta_a
        beta_b = self.beta_b
        A_mu = self.A_mu
        A_var = self.A_var*th.ones_like(A_mu)
       
        if o.reference == '' and "s" in inputs.keys():
            s_drop_rate = o.s_drop_rate if self.training else 0
            if th.rand([]).item() < 1 - s_drop_rate:
                s = inputs["s"]

        # Encode x_m
        z_x_mu, z_x_logvar = {}, {}
        x_pp = {}
        d = {}
        for m in x.keys():
            x_pp[m] = preprocess(x[m], m, o.dims_x[m], o.task)
            
            if m in ["rna", "adt"]:  # use mask
                h = x_pp[m] * e[m]
            elif m == "atac":        # encode each chromosome
                x_chrs = x_pp[m].split(o.dims_chr, dim=1)
                h_chrs = [self.chr_encs[i](x_chr) for i, x_chr in enumerate(x_chrs)]
                h = self.chr_enc_cat_layer(th.cat(h_chrs, dim=1))
            else:
                h = x_pp[m]
            # encoding
            z_x_mu[m], z_x_logvar[m], d_x = self.x_encs[m](h).split([o.dim_z, o.dim_z, o.dim_c], dim=1)

        # Encode s
        if s is not None:
            s_pp = nn.functional.one_hot(s["joint"].squeeze(1), num_classes=o.dim_s).float()  # N * B
            z_s_mu, z_s_logvar, d_s = self.s_enc(s_pp).split([o.dim_z, o.dim_z, o.dim_c], dim=1)
            z_s_mu, z_s_logvar = [z_s_mu], [z_s_logvar]
        else:
            z_s_mu, z_s_logvar = [], []
            d_s = th.zeros_like(d_x)

        # Use product-of-experts
        z_x_mu_list = list(z_x_mu.values())
        z_x_logvar_list = list(z_x_logvar.values())
        z_mu, z_logvar = poe(z_x_mu_list+z_s_mu, z_x_logvar_list+z_s_logvar)  # N * K
    
        d = d_x+d_s
        
        # Sample z
        if self.training:
            z = utils.sample_gaussian(z_mu, z_logvar)
        elif self.sampling and o.sample_num > 0:
            z_mu_expand = z_mu.unsqueeze(1)  # N * 1 * K
            z_logvar_expand = z_logvar.unsqueeze(1).expand(-1, o.sample_num, o.dim_z)  # N * I * K
            z = utils.sample_gaussian(z_mu_expand, z_logvar_expand).reshape(-1, o.dim_z)  # NI * K
        else:  # validation
            z = z_mu

        c, b = z.split([o.dim_c, o.dim_b], dim=1)
        
        # # Generate s activation
        # if s is not None:
        #     e1, e2, s_r_pre = self.s_dec(b).split([A_mu.size(1), A_mu.size(1), o.dim_s], dim=1)
        # else:
        #     s_r_pre = None
        # e1, e2, s_r_pre = self.s_dec(b).split([A_mu.size(1), A_mu.size(1), o.dim_s], dim=1)
        
        # Beta process
        beta_a = beta_a.expand(d.size(0), o.dim_c)
        beta_b = beta_b.expand(d.size(0), o.dim_c)
        hat_beta = Beta(beta_a, beta_b).sample()   
        beta = hat_beta.cumprod(dim=1)*(1-eps) + eps    # sum = 1?
        logit_beta_d = th.logit(beta) + d
        
        # Sample w
        logit_beta_d_1 = logit_beta_d*0.5
        logit_beta_d_2 = logit_beta_d*-0.5

        logit_beta_d_cat = th.stack((logit_beta_d_1, logit_beta_d_2), dim=1)
                 
        post_Ber = logit_beta_d_cat
        prior_Ber = beta 
        
        bin_mat = utils.sample_Gumbel_softmax(post_Ber)
        w = bin_mat[:,0]
        
        # get posterior distributions 
        EW = nn.functional.softmax(logit_beta_d_1, dim=1)
        EA = A_mu
        
        # Loadings
        A_logvar = A_var.log()
        A = utils.sample_gaussian(A_mu, A_logvar)
               
        # Generate x_m activation/probability
        E1 = self.E1
        E2 = self.E2
        bE1 = th.mm(b, E1).exp()
        bE2 = th.mm(b, E2)
        x_r_h = nn.functional.linear(c*w, A)      
        x_r_pre = (x_r_h*bE1 + bE2).split(list(o.dims_h.values()), dim=1)
        
        x_r_pre = utils.get_dict(o.mods, x_r_pre)
                             
        if "atac" in x_r_pre.keys():
            h_chrs = self.chr_dec_split_layer(x_r_pre["atac"]).split(o.dims_dec_chr[0], dim=1)
            x_chrs = [self.chr_decs[i](h_chr) for i, h_chr in enumerate(h_chrs)]
            x_r_pre["atac"] = th.cat(x_chrs, dim=1).sigmoid()
        
        # Generate s activation
        if s is not None:
            s_r_pre = self.s_dec(b)
        else:
            s_r_pre = None  
        #
        z_uni, cw_all = {}, {}
        for m in z_x_mu.keys():
            # Calculate q(z|x^m, s)
            z_uni_mu, z_uni_logvar = poe([z_x_mu[m]]+z_s_mu, [z_x_logvar[m]]+z_s_logvar)  # N * K
            z_uni[m] = utils.sample_gaussian(z_uni_mu, z_uni_logvar)  # N * K
            cw_all[m] = z_uni[m][:, :o.dim_c]*w  # N * C
        cw_all["joint"] = c*w
  
  
        return x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, cw_all, A, alpha, beta_a, beta_b, post_Ber, prior_Ber, w, A_mu, A_logvar, EW, EA


def poe(mus, logvars):
    """
    Product of Experts
    - mus: [mu_1, ..., mu_M], where mu_m is N * K
    - logvars: [logvar_1, ..., logvar_M], where logvar_m is N * K
    """
    
    mus = [th.full_like(mus[0], 0)] + mus
    logvars = [th.full_like(logvars[0], 0)] + logvars
    
    mus_stack = th.stack(mus, dim=1)  # N * M * K
    logvars_stack = th.stack(logvars, dim=1)
    
    T = exp(-logvars_stack)  # precision of i-th Gaussian expert at point x
    T_sum = T.sum(1)  # N * K
    pd_mu = (mus_stack * T).sum(1) / T_sum
    pd_var = 1 / T_sum
    pd_logvar = log(pd_var)
    return pd_mu, pd_logvar  # N * K


def gen_real_data(x_r_pre, sampling=True):
    """
    Generate real data using x_r_pre
    - sampling: whether to generate discrete samples
    """
    x_r = {}
    for m, v in x_r_pre.items():
        if m in ["rna", "adt"]:
            x_r[m] = v.exp()
            x_r[m] = v
            if sampling:
                x_r[m] = th.poisson(x_r[m]).int()
        else:  # for atac
            x_r[m] = v
            if sampling:
                x_r[m] = th.bernoulli(x_r[m]).int()
    return x_r


class Discriminator(nn.Module):

    def __init__(self, o):
        super(Discriminator, self).__init__()
        self.o = o

        predictors = {}
        mods = o.mods + ["joint"]
        for m in mods:
            predictors[m] = MLP([o.dim_c]+o.dims_discriminator+[o.dims_s[m]],
                                    hid_norm=o.norm, hid_drop=o.drop)
        self.predictors = nn.ModuleDict(predictors)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  # log_softmax + nll
        
        self.epoch = None
        
        
    def forward(self, cw_all, s_all):
        o = self.o
        loss_dict = {}
        
        for m in s_all.keys():
            c, s = cw_all[m], s_all[m]
            s_r_pre = self.predictors[m](c)
            loss_dict[m] = self.cross_entropy_loss(s_r_pre, s.squeeze(1))

            if m == "joint":
                prob = s_r_pre.softmax(dim=1)
                mask = nn.functional.one_hot(s.squeeze(1), num_classes=o.dims_s[m])
                self.prob = (prob * mask).sum(1).mean().item()
                           
        loss = sum(loss_dict.values()) / cw_all["joint"].size(0) * 40
        return loss



class LossCalculator(nn.Module):

    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        # self.log_softmax = func("log_softmax")
        # self.nll_loss = nn.NLLLoss(reduction='sum')

        self.pois_loss = nn.PoissonNLLLoss(log_input=True , full=True, reduction='none') 
        # self.pois_loss = nn.PoissonNLLLoss(full=True, reduction='none')
         
        self.bce_loss = nn.BCELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # log_softmax + nll
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.gaussian_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
        # self.enc_s = MLP([o.dim_s]+o.dims_enc_s+[o.dim_b*2], hid_norm=o.norm, hid_drop=o.drop)

    def forward(self, inputs, x_r_pre, s_r_pre, z_mu, z_logvar, b, z_uni, A, alpha, beta_a, beta_b, post_Ber, prior_Ber, w, A_mu, A_logvar, EW, EA):
        o = self.o
        s = inputs["s"]["joint"]
        x = inputs["x"]
        e = inputs["e"]
        prior_alpha = (th.ones(beta_a.size())*alpha).cuda()
        prior_beta = (th.ones(beta_b.size())).cuda()
        
        loss_recon = self.calc_recon_loss(x, s, e, x_r_pre, s_r_pre)
        # loss_jsd_s = self.calc_jsd_s_loss(s_r_pre)
        loss_kld_z = self.calc_kld_z_loss(z_mu, z_logvar)
        loss_A = self.calc_kla_loss(A_mu, A_logvar)
        # loss_beta = self.KL_Beta_Kum(beta_a, beta_b, prior_alpha, prior_beta, log_beta_prior)
        loss_beta = self.KL_Beta(beta_a, beta_b, prior_alpha, prior_beta)
        loss_Ber = self.KL_Ber(post_Ber, prior_Ber)   
        if o.experiment in ["no_kl", "no_kl_ad"]:
            loss_kld_z = loss_kld_z * 0
        # loss_topo = self.calc_topology_loss(x_pp, z_x_mu, z_x_logvar, z_s_mu, z_s_logvar) * 500
        loss_topo = self.calc_consistency_loss(z_uni) * 50
        # loss_mi = self.calc_mi_loss(b, s) * 0
        # l1_regularization = abs(A).sum()
        # l2_regularization = abs(A).pow(2).sum()

        if o.debug == 1:
            print("recon: %.3f\tkld_z: %.3f\ttopo: %.3f\tkl_Ber:%.3f\tkl_beta:%.3f\tkl_A:%.3f" % (loss_recon.item(),
                loss_kld_z.item(), loss_topo.item(), loss_Ber.item(), loss_beta.item(), loss_A.item()))

        return loss_recon + loss_kld_z + loss_topo + loss_Ber + loss_beta + loss_A / s.size(0)
            
    def calc_recon_loss(self, x, s, e, x_r_pre, s_r_pre):
        o = self.o
        losses = {}
        # Reconstruciton losses of x^m
        for m in x.keys():
            if m == "label":
                losses[m] = self.cross_entropy_loss(x_r_pre[m], x[m].squeeze(1)).sum()
            elif m == "atac":
                losses[m] = self.bce_loss(x_r_pre[m], x[m]).sum()
                # losses[m] = self.pois_loss(x_r_pre[m], x[m]).sum()
            else:
                losses[m] = (self.pois_loss(x_r_pre[m], x[m]) * e[m]).sum()
        if s_r_pre is not None:
            losses["s"] = self.cross_entropy_loss(s_r_pre, s.squeeze(1)).sum() * 1000
        # print(losses)
        return sum(losses.values()) / s.size(0)


    def calc_kld_z_loss(self, mu, logvar):
        o = self.o
        mu_c, mu_b = mu.split([o.dim_c, o.dim_b], dim=1)
        logvar_c, logvar_b = logvar.split([o.dim_c, o.dim_b], dim=1)
        kld_c_loss = self.calc_kld_loss(mu_c, logvar_c)
        kld_b_loss = self.calc_kld_loss(mu_b, logvar_b)
        beta = 5
        kld_z_loss = kld_c_loss + beta * kld_b_loss
        return kld_z_loss

    def calc_kld_loss(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / mu.size(0)
    
    def KL_Beta(self, a, b, prior_alpha, prior_beta):
        q_Beta = Beta(a, b)
        p_beta = Beta(prior_alpha, prior_beta)
        kl_Beta = kl_divergence(q_Beta, p_beta)
        return kl_Beta.sum() / len(a) 

    def KL_Ber(self, post_Ber, prior_Ber):
        post_Ber_softmax = nn.functional.softmax(post_Ber, dim=1)
        post_Ber_0 = post_Ber_softmax[:,0] 
        kl_ber = post_Ber_0*log(post_Ber_0/prior_Ber)\
            +(1.0-post_Ber_0)*log((1.0-post_Ber_0)/(1.0-prior_Ber))
        kl = kl_ber.sum() / len(prior_Ber)
        return kl    
    
    # def calc_kla_loss(self, mu, logvar): 
    #     return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() 
    
    def calc_kla_loss(self, mu, logvar): 
        return (-0.5 * (- mu.pow(2))).sum() 

    def calc_consistency_loss(self, z_uni):
        z_uni_stack = th.stack(list(z_uni.values()), dim=0)  # M * N * K
        z_uni_mean = z_uni_stack.mean(0, keepdim=True)  # 1 * N * K
        return ((z_uni_stack - z_uni_mean)**2).sum() / z_uni_stack.size(1)


class MLP(nn.Module):
    def __init__(self, features=[], hid_trans='mish', out_trans=False,
                 norm=False, hid_norm=False, drop=False, hid_drop=False):
        super(MLP, self).__init__()
        layer_num = len(features)
        assert layer_num > 1, "MLP should have at least 2 layers!"
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False
        
        layers = []
        for i in range(1, layer_num):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < layer_num - 1:  # hidden layers (if layer number > 2)
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:                  # output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class Layer1D(nn.Module):
    def __init__(self, dim=False, norm=False, trans=False, drop=False):
        super(Layer1D, self).__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if trans:
            layers.append(func(trans))
        if drop:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def preprocess(x, name, dim, task):
    if name == "label":
        x = nn.functional.one_hot(x.squeeze(1), num_classes=dim).float()
    # elif name == "atac":
    #     x = x.log1p()
    elif name == "rna":
        x = x.log1p()
    elif name == "adt":
        x = x.log1p()
    return x


def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0)  # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)


def exp(x, eps=1e-12):
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x, eps=1e-12):
    return (x + eps).log()


def func(func_name):
    if func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'silu':
        return nn.SiLU()
    elif func_name == 'mish':
        return nn.Mish()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    elif func_name == 'log_softmax':
        return nn.LogSoftmax(dim=1)
    else:
        assert False, "Invalid func_name."


class CheckBP(nn.Module):
    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):
    def forward(self, input):
        return F.Identity.apply(input)
