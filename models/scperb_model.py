import os
import pdb
import torch
import numpy as np
from torch import nn
from scipy import stats
from scipy import sparse
from .scperb_vae import scPerb_vae
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

class scperb(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.criterion = nn.MSELoss(reduction = 'mean')
        self.l1_loss = nn.L1Loss()
        self.Sl1_loss = nn.SmoothL1Loss()
        self.model = scPerb_vae(opt).to(opt.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), opt.lr)
        self.loss_stat = {}

    def set_input(self, con, sti, sty):
        self.con = con.to(self.opt.device)
        self.sti = sti.to(self.opt.device)
        self.sty = sty.to(self.opt.device)
        
    def forward(self):
        self.model.train()
        self.out = self.model(self.con, self.sti, self.sty)
        return self.out
    
    def get_reconstruction_loss(self, x, px):
        loss = ((x - px) ** 2).mean()
        return loss

    def print_res(self, stim):
        a = stim.mean()
        b = stim.max()
        c = stim.min()

        print(a)
        print(b)
        print(c)

    def compute_loss(self, epochs):
        
        z_con, m_con, v_con = self.out['z_con']
        z_sti, m_sti, v_sti = self.out['z_sti']

        # print(z_sti)
        # f = open('./spaperb_sti.txt', 'w')
        # f.write(str(self.out['output'][1]))
        # f.close()
        # import pdb
        # if epochs == 38:
        #     pdb.set_trace()
        
        kld_con = kl(Normal(m_con, torch.sqrt(torch.exp(v_con))),Normal(0, 1)).mean()
        kld_sti = kl(Normal(m_sti, torch.sqrt(torch.exp(v_sti))),Normal(0, 1)).mean()
        # kld_con = kl(Normal(m_con, torch.sqrt(torch.exp(v_con))),Normal(0.042248946, 0.050937362)).mean()
        # kld_sti = kl(Normal(m_sti, torch.sqrt(torch.exp(v_sti))),Normal(0.04193066, 0.0505809)).mean()

        rl_con = self.Sl1_loss(self.out['output'][0], self.con)
        rl_sti = self.Sl1_loss(self.out['output'][1], self.sti)
        rl = self.Sl1_loss(self.out['output'][2], self.sti)
        rl_sty = self.Sl1_loss(self.out['delta'], (z_sti - z_con))
        # rl_sty = self.Sl1_loss(self.out['z_pred'], (z_sti))

        # print("delta:")
        # self.print_res(self.out['delta'])

        # rl_con = self.get_reconstruction_loss(self.out['output'][0], self.con)
        # rl_sti = self.get_reconstruction_loss(self.out['output'][1], self.sti)
        # rl_sty = self.get_reconstruction_loss(self.out['delta'], (z_sti - z_con))
        # rl_cng = self.get_reconstruction_loss(self.out['output'][2], self.sti)

        # rl_con = self.Sl1_loss(self.out['output'][0], self.con)
        # rl_sti = self.Sl1_loss(self.out['output'][1], self.sti)
        # rl_cng = self.Sl1_loss(self.out['output'], self.sti)
        # print(self.out['output'])

        # self.print_res(self.out['output'][2])
        # loss = rl_cng + 0.5 * ((kld_con) * self.opt.alpha)
        # rl_cng = self.criterion(self.out['output'][2], sti).sum(dim = 1)
        # rl_cng = self.Sl1_loss(self.out['output'][2], self.sti)        
        # loss = torch.mean((rl_con + rl_sti) + 0.5 * ((kld_con + kld_sti) * self.opt.alpha) + rl_sty)
        # loss = (0.5 * (rl_con + rl_sti) + 0.5 * ((kld_con + kld_sti) * self.opt.alpha) + self.opt.delta * rl_sty) + self.opt.beta * rl_cng
        # loss = rl_cng + rl_sty

        # print(self.opt.model_use)

        # if self.opt.model_use == '2enc_spaperb' or self.opt.model_use == 'supervise_2enc_spaperb':
        # loss = (0.5 * (rl_con + rl_sti) + 0.5 * ((kld_con + kld_sti) * self.opt.alpha) + rl_sty)
            # print("not used")
        # else:
        loss = rl + 0.5 * (kld_con) * self.opt.alpha + 0.5 * (kld_sti) * self.opt.beta + rl_sty
            # print("use")
        
        self.loss = loss
        # self.loss2 = rl_sty.mean()

        self.loss_stat = {
            # 'kl_con': torch.mean(kld_con * self.opt.alpha),
            # 'kl_sti': torch.mean(kld_sti * self.opt.alpha),
            # 'rl_con': rl_con.mean(),
            # 'rl_sti': rl_sti.mean(),
            # 'rl_cng': rl_cng.mean() * self.opt.beta,
            'tot': self.loss,
            'kl': torch.mean(0.5 * ((kld_con) * self.opt.alpha)) * 100,
            # 'cng': (rl_cng) * 100
            # 'rec': torch.mean((rl_con + rl_sti)) * 100,
            'rl': rl * 100,
            'sty': torch.mean(0.5 * rl_sty) * 100,
            # 'cng': torch.mean(rl_cng)
        }

    def get_current_loss(self):
        return self.loss_stat

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_parameter(self, epochs):
        self.forward()
        self.compute_loss(epochs)
        self.backward()
    
    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).to(self.opt.device)
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X
    
    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))

    def predict(self, con, sty):
        self.model.eval()
        gen_img = self.model.predict(con.to(self.opt.device), sty.to(self.opt.device))
        return self.tensor2numpy(gen_img)

    def predict_new(self, con, sty):
        self.model.eval()
        outs = self.model(con.to(self.opt.device), con.to(self.opt.device), sty.to(self.opt.device))
        # print(outs['output'])
        _, __, gen_img = outs['output']
        return self.tensor2numpy(gen_img)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)