# from options.opt import opt
import torch
import random
import numpy as np
import scanpy as sc
from scipy import sparse
import torch.utils.data as data

class customDataloader(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        train = sc.read(self.opt.read_path)
        train = train[~((train.obs[opt.cell_type_key] == opt.exclude_celltype) & (train.obs[opt.condition_key] == opt.stim_key))]
        # sc.pp.log1p(train)
        # self.train = train[~((train.obs[opt.cell_type_key] == opt.exclude_celltype) & (train.obs[opt.condition_key] == opt.ctrl_key))]

        # valid = sc.read(self.opt.read_valid_path)
        valid = sc.read(self.opt.read_path)
        # sc.pp.log1p(valid)
        self.return_valid = valid
        self.cell_type = valid[valid.obs[opt.cell_type_key] == opt.exclude_celltype]
        stim = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.stim_key))]
        self.valid = valid[~((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs[opt.condition_key] == opt.stim_key))]
        self.pred_data = self.valid[((self.valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (self.valid.obs["condition"] == opt.ctrl_key))]
        self.stim = self.adata2numpy(stim)
        self.ctrl_data = self.valid[self.valid.obs['condition'] == opt.ctrl_key, :]
        self.stim_data = self.valid[self.valid.obs['condition'] == opt.stim_key, :]
        self.len_valid = len(self.pred_data)
        
        train = self.balance(train)
        
        if opt.model_use == 'vae_test1':
            condition_mask = (train.obs[opt.condition_key] == opt.stim_key)
            sti = train[condition_mask].copy()
            con = train[~condition_mask].copy()
            self.len = len(sti)
            self.sti_np = self.adata2tensor(sti)
            self.con_np = self.adata2tensor(con)

        else:
            self.len = len(train)
            self.train_np = self.adata2tensor(train)

    def get_stat(self):
        con_num = self.adata2numpy(self.con)
        sti_num = self.adata2numpy(self.sti)
        con_data = con_num[con_num > 0]
        sti_data = sti_num[sti_num > 0]
        return float(con_data.mean()), float(con_data.var()), float(sti_data.mean()), float(sti_data.var())

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data)
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
    
    def __getitem__(self, idx):
        if self.opt.model_use == 'vae_test1':
            return (self.con_np[idx], self.sti_np[idx])
        else:
            return (self.train_np[idx])

    def __len__(self):
        return self.len

    def get_val_data(self):
        return self.pred_data, self.ctrl_data, self.stim_data, self.stim, self.cell_type

    def balance(self, adata):
        cell_type = adata.obs[self.opt.cell_type_key]

        class_num = np.unique(cell_type)
        type_num = {}
        max_num = -1
        for i in class_num:
            type_num[i] = cell_type[cell_type == i].shape[0]
            max_num = max(max_num, type_num[i])
        
        index_add = []
        for i in class_num:
            class_index = np.array(cell_type == i)
            index_cls = np.nonzero(class_index)[0]
            index_cls = index_cls[np.random.choice(len(index_cls), max_num)]
            index_add.append(index_cls)

        balanced_data = adata[np.concatenate(index_add)].copy()
        return balanced_data