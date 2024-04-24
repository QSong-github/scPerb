# from options.opt import opt
import pdb
import torch
import random
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import sparse
import torch.utils.data as data

class customDataloader(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        train = sc.read(self.opt.read_path)
        
        if opt.supervise == False:
            valid = train
            train = train[~(train.obs[opt.cell_type_key] == opt.exclude_celltype)]
            print("not supervise")
        
        else:
            valid = sc.read(self.opt.read_valid_path)
            train = train[(train.obs[opt.cell_type_key] == opt.exclude_celltype)]
            # print(valid)
            print("supervised")
        # sc.pp.log1p(train)

        
        con, sti = self.balance(train)
        self.len = len(sti)
        # print(len(sti))
        # print(len(con))
        # exit(0)
        self.sti_np = self.adata2tensor(sti)
        self.con_np = self.adata2tensor(con)
        self.size = self.adata2numpy(sti)
        
        
        # valid = sc.read(self.opt.read_valid_path)
        # valid = train
        # sc.pp.log1p(valid)
        # valid = sc.read(self.opt.read_path)
        # print(valid)
        self.cell_type = valid[valid.obs[opt.cell_type_key] == opt.exclude_celltype]
        self.ctrl_data = valid[(valid.obs["condition"] == opt.ctrl_key)]
        self.stim_data = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.stim_key))]
        # print(self.stim_data)
        self.stim = self.adata2numpy(self.stim_data)
        self.stim = self.change_dif(self.stim)
        # print(self.stim)
        self.stim_tensor = self.numpy2tensor(self.stim)
        self.valid = valid[~((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs[opt.condition_key] == opt.stim_key))]
        self.pred_data = self.adata2numpy(self.valid[((self.valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (self.valid.obs["condition"] == opt.ctrl_key))])
        self.pred_data = self.numpy2tensor(self.change_dif(self.pred_data))
        self.stim_len = len(self.stim)
        # print("len", len(self.pred_data))
        self.len_valid = len(self.pred_data)
        # self.stim_data = self.adata2tensor(self.stim_data)
        
        self.sty = torch.rand(self.opt.input_dim)
        # self.sty = torch.ones(self.opt.input_dim)
        # pdb.set_trace()

    def change_dif(self, data):
        dif = data.shape[1] - self.size.shape[1]
        # print(data.shape, self.size.shape,dif)
        if dif<0:
            print("out of range!")
            return None
        for i in range(dif):
            data = np.delete(data,int(random.random() * data.shape[1]),1)
        return data

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
        if self.opt.validation == True:
            return (self.pred_data[idx], self.sty)
        else:
            return (self.con_np[idx], self.sti_np[idx], self.sty)

    def __len__(self):
        if self.opt.validation == True:
            return self.len_valid
        else:
            return self.len

    def get_val_data(self):
        return self.ctrl_data, self.stim_data, self.cell_type
    
    def get_real_stim(self):
        return self.stim

    def balance(self, adata):
        # cell_type = adata.obs[self.opt.cell_type]
        ctrl = adata[adata.obs['condition'] == self.opt.ctrl_key]
        stim = adata[adata.obs['condition'] == self.opt.stim_key]
        ctrl_cell_type = ctrl.obs[self.opt.cell_type_key]
        stim_cell_type = stim.obs[self.opt.cell_type_key]
        class_num = np.unique(ctrl_cell_type)
        max_num = {}
        
        for i in class_num:
            # x = ctrl_cell_type[ctrl_cell_type == i].shape[0]
            # y = stim_cell_type[stim_cell_type == i].shape[0]
            # maxm = max(x, y)
            max_num[i] = (max(ctrl_cell_type[ctrl_cell_type == i].shape[0], stim_cell_type[stim_cell_type == i].shape[0]))
        
        ctrl_index_add = []
        strl_index_add = []

        for i in class_num:
            ctrl_class_index = np.array(ctrl_cell_type == i)
            stim_class_index = np.array(stim_cell_type == i)
            stim_fake = np.ones(len(stim_cell_type))
            
            ctrl_index_cls = np.nonzero(ctrl_class_index)[0]
            stim_index_cls = np.nonzero(stim_class_index)[0]
            stim_fake = np.nonzero(stim_fake)[0]
            # print(stim_fake)

            # print(ctrl_index_cls)
            # print(type(ctrl_index_cls))
            
            # print(max_num[i])
            # print(len(ctrl_index_cls))
            # print(len(stim_index_cls))
            stim_len = len(stim_index_cls)

            if stim_len == 0:
                stim_len = len(stim_cell_type)
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_fake[np.random.choice(stim_len, max_num[i])]
            
            else:
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_index_cls[np.random.choice(stim_len, max_num[i])]
                
            ctrl_index_add.append(ctrl_index_cls)
            strl_index_add.append(stim_index_cls)

        balanced_data_ctrl = ctrl[np.concatenate(ctrl_index_add)]
        balanced_data_stim = stim[np.concatenate(strl_index_add)]

        return balanced_data_ctrl, balanced_data_stim