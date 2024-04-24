import os
import torch
import argparse

class options():      
    def init(self):
        self.opt = self.get_opt()
        self.make_dic()
        self.check_device()
        self.check_data()
        return self.opt
    
    def check_data(self):
        opt = self.opt
        if self.opt.data == 'pbmc':
            opt.stim_key = 'stimulated'
            opt.ctrl_key = 'control'
            opt.cell_type_key = 'cell_type'
            opt.input_dim = 6998

        elif self.opt.data == 'hpoly':
            opt.stim_key = 'Hpoly.Day10'
            opt.ctrl_key = 'Control'
            opt.cell_type_key = 'cell_label'
            opt.input_dim = 7000

        elif self.opt.data == 'salmonella':
            opt.stim_key = 'Salmonella'
            opt.ctrl_key = 'Control'
            opt.cell_type_key = 'cell_label'
            opt.input_dim = 7000

        elif self.opt.data == 'species':
            opt.stim_key = 'LPS6'
            opt.ctrl_key = 'unst'
            opt.cell_type_key = 'species'
            opt.input_dim = 6619

        elif self.opt.data == 'study':
            opt.stim_key = 'stimulated'
            opt.ctrl_key = 'control'
            opt.cell_type_key = 'cell_type'
            opt.input_dim = 7000

        self.opt = opt
            
    def make_dic(self):
        opt = self.opt
        self.get_save_path()
        os.makedirs(opt.save_path, exist_ok=True)
        os.makedirs(opt.model_save_path, exist_ok=True)
        os.makedirs(opt.result_save_path, exist_ok=True)

    def check_device(self):
        if torch.cuda.is_available():
            self.opt.device = 'cuda'

    def get_save_path(self):
        opt = self.opt
        opt.save_path = opt.model_use + '_' + opt.save_path + '/' + opt.data 
        if opt.cross is False:
            opt.read_path = './data/train_' + opt.data + '.h5ad'
            opt.read_valid_path = './data/valid_' + opt.data + '.h5ad'
        opt.model_save_path = './' + opt.save_path + '/model_' + opt.model_name
        opt.result_save_path = './' +opt.save_path + f'/res_{opt.model_name}'
        self.opt = opt

    def get_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="scPerb", help="The name of the current model.")
        parser.add_argument("--training", type=bool, default=True, help="Whether training or not.")
        parser.add_argument("--resume", type=bool, default=False)
        parser.add_argument("--data", type=str, default='pbmc', help='Which data to use.')
        parser.add_argument("--cross", type = bool, default = False, help = "Whether cross data or not.")
        parser.add_argument("--read_path", type = str, default = './data/train_pbmc.h5ad', help = "The path of the data.")
        parser.add_argument("--read_valid_path", type = str, default = './data/valid_pbmc.h5ad', help = "The path of the data.")
        parser.add_argument("--backup_url", type = str, default = 'https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
        parser.add_argument("--save_path", type=str, default='saved', help="The folder that stores the saved things.")
        parser.add_argument("--loss_save_path", type=str,default= './saved/saved_loss', help = "The path to save the loss.")
        parser.add_argument("--model_save_path", type = str, default = './saved/saved_model', help = "The path of saved model")
        parser.add_argument("--picture_save_path", type = str, default = './saved/saved_picture', help = "The path of saved picture")
        parser.add_argument("--result_save_path", type = str, default = './saved/result', help = "The path of saved result")
        parser.add_argument("--save_log", type=str, default='saved/saved_log', help='The path to save the log.')
        
        parser.add_argument("--batch_size", type = int, default = 256, help = "This is the batch size.")
        parser.add_argument("--num_workers", type = int, default = 0, help= "How many cpus will try to load the data into the model.")
        parser.add_argument("--lr", type = float, default = 1e-3, help = "This is the learning rate.")

        parser.add_argument("--style_dim", type = int, default = 1, help = "This is the size of the style layer.")
        parser.add_argument("--latent_dim", type = int, default = 100, help = "This is the size of the context hidden layer")
        parser.add_argument("--context_latent_dim", type = int, default = 100, help = "This is the size of the context hidden layer")
        parser.add_argument("--style_latent_dim", type = int, default = 1, help = "This is the size of the context hidden layer")
        parser.add_argument("--hidden_dim", type = int, default = 800, help = "This is the size of the latent layer.")
        parser.add_argument("--input_dim", type = int, default = 6998, help = "This is the size of the input layer.")
        parser.add_argument("--drop_out", type = float, default = 0.2, help = "This is the drop out rate.")
        parser.add_argument("--alpha", type = float, default = 0.01, help = "This is the parameter before KLD loss")
        parser.add_argument("--beta", type = float, default = 0, help = "This is the parameter before KLD loss")
        parser.add_argument("--delta", type = float, default=1, help="This is the parameter of style loss.")
        parser.add_argument("--epochs", type = int, default = 500, help = "This is the number of epochs.")
        
        parser.add_argument("--save_interval", type = int, default = 20, help = "Save model every how many epochs.")
        parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the input data or not.")
        
        parser.add_argument("--get_epoch", type=int, default=0, help="Which model to load.")
        parser.add_argument("--stim_key", type=str, default="stimulated", help="This is the stimulation key.")
        parser.add_argument("--pred_key", type=str, default="pred", help="This is the prediction key.")
        parser.add_argument("--ctrl_key", type=str, default="control", help="This is the control key.")
        parser.add_argument("--condition_key", type=str, default="condition", help="This is the condition key.")
        parser.add_argument("--cell_type_key", type = str, default="cell_type", help="This is the cell type key.")
        parser.add_argument("--exclude_celltype", type=str, default="CD4T", help="The type of the cell that is going to be excluded.")
        parser.add_argument("--plot_only", type=bool, default=False, help="Whether it is validation or not.")
        parser.add_argument("--plot", type=bool, default=False, help="Whether to plot or not.")

        parser.add_argument("--device", type = str, default = 'cpu', help = "Which device to use.")

        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument("--use_model", type=str, default='all', help="Wheter to use the best model to predict or not.")
        parser.add_argument("--model_use", type=str, default='scperb', help="Wheter to use the best model to predict or not.")
        parser.add_argument("--download_data", type=bool, default=False, help="Whether to download data or not.")
        parser.add_argument("--validation", type=bool, default=False, help="Whether this is validation or not.")

        parser.add_argument("--supervise", type = bool, default = False, help = "Whether to put in stimulus or not.")
        self.opt = parser.parse_args()
        return self.opt