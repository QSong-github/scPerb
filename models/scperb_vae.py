import torch
from torch import nn

class scPerb_vae(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        input_dim = opt.input_dim
        hidden_dim = opt.hidden_dim
        latent_dim = opt.latent_dim
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
        )
        
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU()
            # nnx = self.encoder2(sty).Sigmoid()
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, input_dim),
        )
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # eps = 1
        return mu + eps*std 

    def get_z(self, x):
        hidden = self.encoder1(x)
        mu = self.mu_encoder(hidden)
        logvar = self.logvar_encoder(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def print_res(self, stim):
        a = stim.mean()
        b = stim.max()
        c = stim.min()

        print(a)
        print(b)
        print(c)

    def forward(self, con, sti, sty):
        z_con, mu_con, logvar_con = self.get_z(con)
        z_sti, mu_sti, logvar_sti = self.get_z(sti)

        # print("z_con:")
        # self.print_res(z_con)

        x = self.encoder2(sty)

        outs = {
            'z_con': [z_con, mu_con, logvar_con],
            'z_sti': [z_sti, mu_sti, logvar_sti],
            'z_pred': x + z_con,
            'output': [self.decoder(z_con), self.decoder(z_sti), self.decoder(x + z_con)],
            'delta': x,
        }
        return outs

    def predict(self, con, sty):
        z_con, mu_con, logvar_con = self.get_z(con)
        
        # print("pred:")
        # self.print_res(z_con)
        # print("in model")
        # print(sty)
        x = self.encoder2(sty)
        # print("x:")
        # self.print_res(x)

        return self.decoder(z_con + x)