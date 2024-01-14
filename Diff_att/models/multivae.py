
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):

    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)



class MultiVAE(nn.Module):

    def __init__(self,rating_matrix,device):
        super(MultiVAE, self).__init__( )

        self.layers = [600]
        self.device = device
        self.lat_dim = 148
        self.drop_out = 0.5
        self.anneal_cap = 0.2
        self.total_anneal_steps = 200000
        self.rating_matrix = rating_matrix.to(self.device)
        self.n_items = rating_matrix.shape[1]
    
        self.update = 0
        
        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        #self.decode_layer_dims是一个list，包含了每一层的输出维度
        
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)
        # parameters initialization
        self.apply(xavier_normal_initialization)
        

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = h.to(self.device)
        h = self.encoder(h)

        #mu = h[:, : int(self.lat_dim / 2)]
        #logvar = h[:, int(self.lat_dim / 2) :]
        mu = h[:, : int(74)]
        logvar = h[:, int(74) :]
       
        z = self.reparameterize(mu, logvar)
        z_1 = z
        #print(z.shape)
        z = self.decoder(z)
        return z, mu, logvar,z_1

    def calculate_loss(self,batch_rating_matrix):
        batch_rating_matrix = batch_rating_matrix.to(self.device)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar,z_1 = self.forward(batch_rating_matrix)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * batch_rating_matrix).sum(1).mean()

        return ce_loss + kl_loss,z_1

    def calculate_loss1(self,batch_rating_matrix):
        batch_rating_matrix = batch_rating_matrix.to(self.device)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z_1 = self.forward1(batch_rating_matrix)

        return z_1

    def calculate_loss2(self,batch_rating_matrix,q):
        batch_rating_matrix = batch_rating_matrix.to(self.device)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, mu, logvar = self.forward2(batch_rating_matrix,q)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * batch_rating_matrix).sum(1).mean()

        return ce_loss + kl_loss
    
    def forward2(self, rating_matrix,q):
        z = self.decoder(q)
        return z, self.mu, self.logvar

    def forward1(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = h.to(self.device)
        h = self.encoder(h)

        #mu = h[:, : int(self.lat_dim / 2)]
        #logvar = h[:, int(self.lat_dim / 2) :]
        mu = h[:, : int(64)]
        logvar = h[:, int(64) :]
       
        self.mu = mu
        self.logvar = logvar
        z = self.reparameterize(mu, logvar)
        z_1 = z
        return z_1

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores.view(-1)
