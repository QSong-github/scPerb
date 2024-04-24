import torch
import random
import subprocess
import numpy as np
from tqdm import tqdm
from scipy import stats
import anndata
import scanpy as sc
from utils.utils import Utils
from options.option import options
from models.scperb_model import scperb
from scipy import sparse
from dataloader.scperbDataset import customDataloader

def adata2numpy(adata):
    if sparse.issparse(adata.X):
        return adata.X.A
    else:
        return adata.X

def print_res(stim):
    a = stim.mean()
    var = stim.var()
    b = stim.max()
    c = stim.min()
    mid = np.median(stim)

    print("mean:", a)
    print("max:", b)
    print("min:", c)
    print("var:", var)
    print("median", mid)

def validation(opt, model, get_result = False):
    opt.validation = True
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = False, pin_memory = True)
    pred = np.empty((0, opt.input_dim))
    model.to(opt.device, non_blocking=True)
    for idx, (con, sty) in enumerate(dataloader):
        model.eval()
        eval = model.predict(con, sty)
        # print(eval.shape)
        eval[eval<0] = 0
        pred = np.append(pred, eval, axis = 0)
    
    if get_result:
        return pred
    
    # print_res(pred)

    stim_data = dataset.get_real_stim()

    x = np.asarray(np.mean(stim_data, axis=0)).ravel()
    y = np.asarray(np.mean(pred, axis=0)).ravel()

    m, b, r_value_mean, p_value, std_err = stats.linregress(x, y)

    x = np.asarray(np.var(stim_data, axis=0)).ravel()
    y = np.asarray(np.var(pred, axis=0)).ravel()

    m, b, r_value_var, p_value, std_err = stats.linregress(x, y)
    opt.validation = False
    return (((r_value_mean ** 2) * 3) + r_value_var ** 2) * 25, r_value_mean ** 2 * 100, r_value_var ** 2 * 100

def train_model(opt, dataset):
    utils = Utils(opt)

    model = scperb(opt)
    
    if opt.resume == True:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    
    model.to(opt.device, non_blocking=True)
    
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True)

    scores = -1
    best_model = 0
    pbar = tqdm(total=opt.epochs)
    for epoch in range(opt.epochs):
        loss = {}
        for idx, (con, sti, sty) in enumerate(dataloader):
            model.train()
            model.set_input(con, sti, sty)
            model.update_parameter(epoch)
            loss_dic = utils.get_loss(model)
            for (k, v) in loss_dic.items():
                if idx == 0:
                    loss[k] = v
                else:
                    loss[k] += v

        model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
        if epoch % 10 == 0:
            tmp_scores, mean, var = validation(opt, model)
        best_model, scores = utils.bestmodel(scores, tmp_scores, epoch, best_model, model)
        
        utils.update_pbar(loss, scores, best_model, pbar, mean, var, tmp_scores)

def fix_seed(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.seed)

def get_res(opt, model_type = "best"):
    model = scperb(opt)
    
    if model_type == "best":
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_best_epoch.pt')
    else:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    
    predicts = validation(opt, model, True)
    valid = sc.read(opt.read_valid_path)
    pred = anndata.AnnData(predicts, obs={opt.condition_key: [opt.pred_key] * len(predicts), opt.cell_type_key: [opt.exclude_celltype] * len(predicts)}, var={"var_names": valid.var_names})
    if model_type == 'best':
        pred.write_h5ad(opt.result_save_path + "/best_epoch.h5ad")
    else: 
        pred.write_h5ad(opt.result_save_path + "/now_epoch.h5ad")

if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    print(opt)
    fix_seed(opt)
    dataset = customDataloader(opt)
    
    if opt.download_data == True:
        command = "python3 DataDownloader.py"
        subprocess.call([command], shell=True)
    
    if opt.validation == True:
        get_res(opt)
    
    else:
        train_model(opt, dataset)
        get_res(opt)