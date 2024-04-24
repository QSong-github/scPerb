import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import anndata
import seaborn as sns
from scipy import sparse
from scipy import stats
# from models.scgen_model import SCGEN
# from models.gimVi_model import gimVi_model
# from dataloader.testDataset import test_dataset

class Utils():
    def __init__(self, opt):
        self.opt = opt
        # matplotlib.rc('ytick', labelsize=18)
        # matplotlib.rc('xtick', labelsize=18)
        # sc.set_figure_params(dpi_save=300)
        # sc.settings.figdir = opt.picture_save_path

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X

    def reg_plot(
        self,
        axs,
        adata,
        axis_keys,
        labels,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        type='mean',
        **kwargs,
    ):

        sns.set()
        sns.set(color_codes=True)

        condition_key = self.opt.condition_key

        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["y"]])
        ctrl = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["x"]])
        print(stim)
        print(ctrl)
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]])
            ctrl_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]])

            if type == 'variance':

                x_diff = np.asarray(np.var(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.var(stim_diff, axis=0)).ravel()
            else: 
                x_diff = np.asarray(np.mean(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.mean(stim_diff, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        
        if type == 'variance':
            x = np.asarray(np.var(ctrl, axis=0)).ravel()
            y = np.asarray(np.var(stim, axis=0)).ravel()
        else:
            x = np.asarray(np.mean(ctrl, axis=0)).ravel()
            y = np.asarray(np.mean(stim, axis=0)).ravel()
        
        m, b, r_value, p_value, std_err = stats.linregress(x, y)

        if verbose:
            print("All genes var: ", r_value**2)
        
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax = axs)
        ax.tick_params(labelsize=fontsize)
        
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        
        if "y1" in axis_keys.keys():
            if type == 'variance':
                y1 = np.asarray(np.var(self.adata2numpy(real_stim), axis=0)).ravel()
            else:
                y1 = np.asarray(np.mean(self.adata2numpy(real_stim), axis=0)).ravel()
            ax.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                ax.text(x_bar, y_bar, i, fontsize=11, color="black")
                ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    ax.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        
        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        if title is None:
            ax.set_title("", fontsize=12)
        
        else:
            ax.set_title(title, fontsize=12)
        
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.4f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.4f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        return r_value**2

    def make_plots(self, adata, conditions, model_name, axs, x_coeff=0.3, y_coeff=0.1):

        if model_name == "RealCD4T":
            mean_labels = {"x": "ctrl mean", "y": "stim mean"}
            var_labels = {"x": "ctrl var", "y": "stim var"}
        else:
            mean_labels = {"x": "pred mean", "y": "stim mean"}
            var_labels = {"x": "pred var", "y": "stim var"}

        scores = self.reg_plot(     
                                    axs = axs[0,0],
                                    adata=adata, 
                                    axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
                                    gene_list=self.diff_genes[:5],
                                    top_100_genes=self.diff_genes,
                                    # path_to_save=os.path.join(self.opt.picture_save_path, f"Fig{num}_{figure}_{model_name}_reg_mean.pdf"),
                                    legend=False,
                                    title="",
                                    labels=mean_labels,
                                    fontsize=10,
                                    x_coeff=x_coeff,
                                    y_coeff=y_coeff,
                                    show=False,
                                    type = 'mean',
        )
        self.reg_plot(
                                    axs = axs[0,1],
                                    adata=adata, 
                                    axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
                                    gene_list=self.diff_genes[:5],
                                    top_100_genes=self.diff_genes,
                                    # path_to_save=os.path.join(self.opt.picture_save_path, f"Fig{figure}_{model_name}_reg_var.pdf"),
                                    legend=False,
                                    labels=var_labels,
                                    title="",
                                    fontsize=10,
                                    x_coeff=x_coeff,
                                    y_coeff=y_coeff,
                                    # save=True,
                                    type = 'variance',
                                    show=False
        )
        sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
        sc.tl.umap(adata, min_dist=1.1)
        plt.style.use('default')
        sc.pl.umap(adata, color=["condition"],
            # legend_loc=False,
            frameon=False,
            title="",
            palette=matplotlib.rcParams["axes.prop_cycle"],
            show=False,
            ax = axs[1,0],
            # save = True,
            # save_path = 'use.png'
            )
        return scores * 100.0

    def calc_R2(self, adata, cell_type, n_genes=6998, conditions=None):
        if n_genes != adata.shape[1]:
            adata_cell = adata[adata.obs["cell_label"] == cell_type]
            print(adata_cell.obs["condition"].unique().tolist())
            sc.tl.rank_genes_groups(adata_cell, groupby="condition", n_genes=n_genes, method="wilcoxon")
            diff_genes = adata_cell.uns["rank_genes_groups"]["names"][conditions["real_stim"]].tolist()[:n_genes//2] \
                    + adata_cell.uns["rank_genes_groups"]["names"][conditions["ctrl"]].tolist()[:n_genes//2]
            adata = adata[:, diff_genes]
        r_values = np.zeros((1, 100))
        real_stim = adata[adata.obs["condition"] == conditions["real_stim"]]
        pred_stim = adata[adata.obs["condition"] == conditions["pred_stim"]]
        for i in range(100):
            pred_stim_idx = np.random.choice(range(0, pred_stim.shape[0]), int(0.8 * pred_stim.shape[0]))
            real_stim_idx = np.random.choice(range(0, real_stim.shape[0]), int(0.8 * real_stim.shape[0]))
            if sparse.issparse(pred_stim.X):
                pred_stim.X = pred_stim.X.A
                real_stim.X = real_stim.X.A
            x = np.average(pred_stim.X[pred_stim_idx], axis=0)
            y = np.average(real_stim.X[real_stim_idx], axis=0)
            m, b, r_value, p_value, std_err = stats.linregress(x, y)
            r_values[0, i] = r_value ** 2
        return r_values.mean(), r_values.std()

    def get_loss(self, model):
        loss_dic = model.get_current_loss()
        loss_dic['it'] = 1
        return loss_dic

    def bestmodel(self, scores, tmp_scores, epoch, best_model, model):
        if tmp_scores > scores:
            scores = tmp_scores
            best_model = epoch
            model.save(self.opt.model_save_path + '/'  + self.opt.exclude_celltype + '_best_epoch.pt')

        return best_model, scores

    # def plotmodel(self, ctrl_adata, stim_adata, predicts, cell_type_data):
    #     opt = self.opt

    #     if opt.use_model == 'all':
    #         opt.use_model = 'best'
    #         self.plot_result(ctrl_adata, stim_adata, predicts, cell_type_data)
    #         opt.use_model = 'now'
    #         self.plot_result(ctrl_adata, stim_adata, predicts, cell_type_data)
        
    #     else:
    #         self.plot_result(ctrl_adata, stim_adata, predicts, cell_type_data)
    
    def plot(self, data, axs):
        conditions = {"real_stim": self.opt.stim_key, "pred_stim": self.opt.pred_key}
        return self.make_plots(adata = data, conditions = conditions, model_name=self.opt.model_name, x_coeff=0.45, y_coeff=0.8, axs = axs)

    def print_res(self, stim):
        a = stim.mean()
        var = stim.var()
        b = stim.max()
        c = stim.min()

        print(a)
        print(b)
        print(c)
        print(var)

    def plot_result(self, ctrl_adata, stim_adata, predicts, cell_type_data, model_use):

        self.print_res(self.adata2numpy(stim_adata))

        opt = self.opt

        fig, axs = plt.subplots(2, 2, figsize = (10, 10))

        cell_type = opt.exclude_celltype
        # cell_type_data = train[train.obs[opt.cell_type_key] == cell_type]

        pred = anndata.AnnData(predicts, obs={opt.condition_key: [opt.pred_key] * len(predicts), opt.cell_type_key: [cell_type] * len(predicts)}, var={"var_names": cell_type_data.var_names})

        # ctrl_adata = train[((train.obs[opt.cell_type_key] == cell_type) & (train.obs['condition'] == opt.ctrl_key))]
        # stim_adata = train[((train.obs[opt.cell_type_key] == cell_type) & (train.obs['condition'] == opt.stim_key))]
        eval_adata = ctrl_adata.concatenate(stim_adata, pred)

        if model_use == 'best':
            eval_adata.write_h5ad(opt.model_save_path + "/best_epoch.h5ad")
        else: 
            eval_adata.write_h5ad(opt.model_save_path + "/now_epoch.h5ad")
        # print(cell_type_data.shape)
        # exclude_cell = train[train.obs[opt.cell_type_key] == cell_type]
        sc.tl.rank_genes_groups(cell_type_data, groupby="condition", method="wilcoxon")
        self.diff_genes = cell_type_data.uns["rank_genes_groups"]["names"][opt.stim_key]

        tmp = eval_adata[eval_adata.obs['condition'] == self.opt.stim_key]
        self.print_res(self.adata2numpy(tmp))

        sc.pl.violin(eval_adata, keys=self.diff_genes[0], groupby="condition", ax = axs[1,1])
        print(self.diff_genes[0])

        self.plot(eval_adata, axs)
        
        if model_use == 'best':
            fig.savefig(opt.result_save_path + '/best_epoch.pdf')
            print("saved at: ", opt.result_save_path + '/best_epoch.pdf')
        else: 
            fig.savefig(opt.result_save_path + '/now_epoch.pdf')
            print("saved at: ", opt.result_save_path + '/now_epoch.pdf')

        

    def update_pbar(self, loss_dic, scores, best_model, pbar, mean, var, tmp_scores,update = True):
        _, it = loss_dic.popitem()
        loss = {}

        for (k,v) in loss_dic.items():
            loss[k] = "%.2f"%(v/it)

        pbar.set_postfix(loss = loss, score = "%.2f"%(scores), best_model = best_model, mean = mean, var = var, now = tmp_scores)
        
        if update:
            pbar.update()

    # def reg_plot(
    #     self,
    #     axs,
    #     adata,
    #     axis_keys,
    #     labels,
    #     gene_list=None,
    #     top_100_genes=None,
    #     show=False,
    #     legend=True,
    #     title=None,
    #     verbose=False,
    #     x_coeff=0.30,
    #     y_coeff=0.8,
    #     fontsize=14,
    #     type='mean',
    #     **kwargs,
    # ):

    #     sns.set()
    #     sns.set(color_codes=True)

    #     condition_key = "condition"

    #     sc.tl.rank_genes_groups(
    #         adata, groupby=condition_key, n_genes=100, method="wilcoxon"
    #     )
    #     diff_genes = top_100_genes
    #     stim = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["y"]])
    #     ctrl = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["x"]])
        
    #     if diff_genes is not None:
    #         if hasattr(diff_genes, "tolist"):
    #             diff_genes = diff_genes.tolist()
    #         adata_diff = adata[:, diff_genes]
    #         stim_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]])
    #         ctrl_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]])

    #         if type == 'variance':

    #             x_diff = np.asarray(np.var(ctrl_diff, axis=0)).ravel()
    #             y_diff = np.asarray(np.var(stim_diff, axis=0)).ravel()
    #         else: 
    #             x_diff = np.asarray(np.mean(ctrl_diff, axis=0)).ravel()
    #             y_diff = np.asarray(np.mean(stim_diff, axis=0)).ravel()
    #         m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
    #             x_diff, y_diff
    #         )
    #         if verbose:
    #             print("Top 100 DEGs var: ", r_value_diff**2)
        
    #     if "y1" in axis_keys.keys():
    #         real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        
    #     if type == 'variance':
    #         x = np.asarray(np.var(ctrl, axis=0)).ravel()
    #         y = np.asarray(np.var(stim, axis=0)).ravel()
    #     else:
    #         x = np.asarray(np.mean(ctrl, axis=0)).ravel()
    #         y = np.asarray(np.mean(stim, axis=0)).ravel()
        
    #     m, b, r_value, p_value, std_err = stats.linregress(x, y)

    #     if verbose:
    #         print("All genes var: ", r_value**2)
        
    #     df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    #     ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax = axs)
    #     ax.tick_params(labelsize=fontsize)
        
    #     if "range" in kwargs:
    #         start, stop, step = kwargs.get("range")
    #         ax.set_xticks(np.arange(start, stop, step))
    #         ax.set_yticks(np.arange(start, stop, step))
        
    #     ax.set_xlabel(labels["x"], fontsize=fontsize)
    #     ax.set_ylabel(labels["y"], fontsize=fontsize)
        
    #     if "y1" in axis_keys.keys():
    #         if type == 'variance':
    #             y1 = np.asarray(np.var(self.adata2numpy(real_stim), axis=0)).ravel()
    #         else:
    #             y1 = np.asarray(np.mean(self.adata2numpy(real_stim), axis=0)).ravel()
    #         ax.scatter(
    #             x,
    #             y1,
    #             marker="*",
    #             c="grey",
    #             alpha=0.5,
    #             label=f"{axis_keys['x']}-{axis_keys['y1']}",
    #         )
        
    #     if gene_list is not None:
    #         for i in gene_list:
    #             j = adata.var_names.tolist().index(i)
    #             x_bar = x[j]
    #             y_bar = y[j]
    #             ax.text(x_bar, y_bar, i, fontsize=11, color="black")
    #             ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
    #             if "y1" in axis_keys.keys():
    #                 y1_bar = y1[j]
    #                 ax.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        
    #     if legend:
    #         ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
    #     if title is None:
    #         ax.set_title("", fontsize=12)
        
    #     else:
    #         ax.set_title(title, fontsize=12)
        
    #     ax.text(
    #         max(x) - max(x) * x_coeff,
    #         max(y) - y_coeff * max(y),
    #         r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.4f}",
    #         fontsize=kwargs.get("textsize", fontsize),
    #     )
        
    #     if diff_genes is not None:
    #         ax.text(
    #             max(x) - max(x) * x_coeff,
    #             max(y) - (y_coeff + 0.15) * max(y),
    #             r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
    #             + f"{r_value_diff ** 2:.4f}",
    #             fontsize=kwargs.get("textsize", fontsize),
    #         )

    #     return r_value**2

