import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
import os
# from core.encoders import *

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
from torch_geometric.nn import global_add_pool

import sys
import json
from torch import optim

from gin_1 import Encoder, FCNet
from utils import *

from args import arg_parse

class FF(nn.Module):
    def __init__(self, input_dim):
        super(FF, self).__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)



class deepGCFX(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers):
        super(deepGCFX, self).__init__()

        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers,summary_iter=2)
        self.fusioner = FCNet([hidden_dim*2, hidden_dim])
        self.eps = 1e-8

        '''self.fusioner = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
        )'''

        self.g_accum = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        '''self.g_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.n_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )'''

        self.g_proj = FF(hidden_dim)
        self.n_proj = FF(hidden_dim)

        self.bn_class = torch.nn.BatchNorm1d(hidden_dim)

        self.mse_loss = nn.MSELoss()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):


        n_nodes = x.size(0)

        # batch_size = data.num_graphs
        #our model
        internal_loss_items, node_mu, node_logvar,gnode_mu, gnode_logvar, graph_mu, graph_logvar = \
            self.encoder.forward_diff(x, edge_index, batch)
        #sum all layers of gnn for summary makes
        '''internal_loss_items, node_mu, node_logvar,gnode_mu, gnode_logvar, graph_mu, graph_logvar = \
            self.encoder.forward_tot4summary_diff(x, edge_index, batch)'''

        '''grouped_mu, grouped_logvar = accumulate_group_evidence(
            graph_mu.data, graph_mu.data, batch, True
        )'''

        node_kl_divergence_loss = (-0.5 / n_nodes) * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))

        graph_kl_divergence_loss = (-0.5 / num_graphs) * torch.mean(torch.sum(
            1 + 2 * graph_logvar - graph_mu.pow(2) - graph_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = node_kl_divergence_loss

        graph_kl_divergence_loss = 0.1*graph_kl_divergence_loss

        node_latent_embeddings_unnorm = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        gnode_latent_embeddings_unnorm = reparameterize(training=True, mu=gnode_mu, logvar=gnode_logvar)
        graph_latent_embeddings_unnorm = reparameterize(training=True, mu=graph_mu, logvar=graph_logvar)
        '''graph_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
        )'''
        node_latent_embeddings = node_latent_embeddings_unnorm
        graph_latent_embeddings = graph_latent_embeddings_unnorm

        _, count = torch.unique(batch,  return_counts=True)

        graph_embedding_expanded = torch.repeat_interleave(graph_latent_embeddings, count, dim=0)

        gl_nodes = self.fusioner(torch.cat([graph_embedding_expanded,node_latent_embeddings], -1))
        #g_nodes = self.fusioner(torch.cat([torch.zeros_like(node_latent_embeddings), graph_latent_embeddings], -1))

        full_node_feat_gl = gl_nodes


        edge_recon_loss_gl = self.recon_loss1(full_node_feat_gl, edge_index, batch)
        edge_recon_loss_g = self.recon_loss1(gnode_latent_embeddings_unnorm, edge_index, batch)

        '''contrastive_loss = self.node_graph_contrastive_loss(self.g_proj(graph_latent_embeddings), self.n_proj(gnode_latent_embeddings_unnorm),
                                                            self.n_proj(node_latent_embeddings_unnorm), batch)'''



        #loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl + contrastive_loss + edge_recon_loss_g
        loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl  + edge_recon_loss_g

        loss.backward()

        return  node_kl_divergence_loss.item(), graph_kl_divergence_loss.item(), 0, \
                edge_recon_loss_gl.item(),edge_recon_loss_g.item()

    def min_max_normalize(self, x, dim=1):
        min_x = torch.min(x, dim, keepdim=True)[0]
        max_x = torch.max(x, dim, keepdim=True)[0]
        range = (max_x - min_x) + self.eps

        normalized = (x- min_x)/range


        return normalized
    def twin_loss(self, g1,g2):
        # empirical cross-correlation matrix
        c = self.bn_class(g1).T @ self.bn_class(g2)


        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1e-2)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(1e-2)
        loss = on_diag + 0.01 * off_diag

        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def recon_loss(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        a, idx_tensor = to_dense_batch(z, batch)
        a_t = a.permute(0, 2, 1)

        rec = torch.bmm(a, a_t)

        org_adj = to_dense_adj(edge_index, batch)


        #pos_weight = float(z.size(0) * z.size(0) - org_adj.sum()) / org_adj.sum()
        #norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - org_adj.sum()) * 2)

        #loss = norm * F.binary_cross_entropy_with_logits(rec, org_adj, pos_weight=pos_weight)
        loss = F.binary_cross_entropy_with_logits(rec, org_adj)
        return loss

    def edge_recon(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss1(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #org_adj = to_dense_adj(edge_index, batch)
        pos_weight = float(z.size(0) * z.size(0) - edge_index.size(0)) / edge_index.size(0)
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - edge_index.size(0)) * 2)



        recon_adj = self.edge_recon(z, edge_index)


        pos_loss = -torch.log(
            recon_adj + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
        neg_loss = -torch.log(1 -
                              self.edge_recon(z, neg_edge_index) +
                              EPS).mean()

        return (pos_loss + neg_loss)

        #loss = F.binary_cross_entropy_with_logits(rec, org_adj)

        #return loss

    def get_positive_expectation(self, p_samples, measure, average=True):
        """Computes the positive part of a divergence / difference.
        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = np.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(- p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples

        if average:
            return Ep.mean()
        else:
            return Ep


    # Borrowed from https://github.com/fanyun-sun/InfoGraph
    def get_negative_expectation(self, q_samples, measure, average=True):
        """Computes the negative part of a divergence / difference.
        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.
        Returns:
            torch.Tensor
        """
        log_2 = np.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples

        if average:
            return Eq.mean()
        else:
            return Eq

    '''def node_graph_contrastive_loss1(self, rep, positive, negative, measure='JSD'):

        pos = (rep * positive).sum(dim=1)
        neg = (rep * negative).sum(dim=1)

        E_pos = self.get_positive_expectation(pos, measure)
        E_neg = self.get_negative_expectation(neg, measure)
        return E_neg - E_pos'''

    def node_graph_contrastive_loss(self, rep, positive, negative, batch, measure='JSD'):

        #num_graphs = rep.shape[0]
        num_nodes = positive.shape[0]


        pos_mask = F.one_hot(batch)
        #neg_mask = torch.ones((num_nodes, num_graphs)).cuda() - pos_mask

        res_positive = torch.mm(positive, rep.t())
        res_negative = torch.mm(negative, rep.t())

        E_pos = self.get_positive_expectation(res_positive * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(res_negative * pos_mask, measure, average=False).sum()
        E_neg = E_neg / num_nodes


        return E_neg - E_pos



if __name__ == '__main__':

    args = arg_parse()

    epochs = args.num_epochs
    model_save_name = args.model_name

    #for seed in [123,132,213,231,321,312]:
    for seed in [123]:
        #for seed in [1234,0,1]:
        #for seed in [32,42,52,62,72,82]:

        print('seed ', seed, 'epochs ', epochs)


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        print('init seed, seed ', torch.initial_seed(), seed)


        losses = {'n_recon_gl':[], 'e_recon_gl':[], 'n_recon_g':[], 'e_recon_g':[], 'node_kl': [],
                  'graph_kl': [], 'contras':[], 'twin':[]}

        warmup_steps = 0
        batch_size = args.batch_size
        lr = args.lr

        EPS = 1e-15

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.DS)
        train_dataset = TUDataset(path, name=args.DS).shuffle()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            dataset_num_features = train_dataset.num_features
        except:
            dataset_num_features = 1

        if not dataset_num_features:

            dataset_num_features = 1

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)


        model = deepGCFX(args.hidden_dim, args.num_gc_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=10, min_lr=0.00001)



        print('================')
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')


        model.train()
        best_recon_loss = 10e40
        for epoch in range(1, epochs+1):
            node_recon_loss_gl_all = 0
            edge_recon_loss_gl_all = 0
            node_recon_loss_g_all = 0
            edge_recon_loss_g_all = 0
            node_kl_loss_all = 0
            graph_kl_loss_all = 0
            contras_loss_all = 0
            twin_loss_all = 0
            for data in train_dataloader:
                data = data.to(device)

                if not train_dataset.num_features:
                    data.x = torch.ones((data.batch.shape[0], dataset_num_features)).to(device)

                optimizer.zero_grad()

                node_kl_divergence_loss, graph_kl_divergence_loss, node_recon_loss_gl, \
                edge_recon_loss_gl, edge_recon_loss_g = model(data.x, data.edge_index, data.batch, data.num_graphs)
                node_recon_loss_gl_all += node_recon_loss_gl
                edge_recon_loss_gl_all += edge_recon_loss_gl
                node_kl_loss_all += node_kl_divergence_loss
                graph_kl_loss_all += graph_kl_divergence_loss
                edge_recon_loss_g_all += edge_recon_loss_g
                optimizer.step()

            losses['n_recon_gl'].append(node_recon_loss_gl_all/ len(train_dataloader))
            losses['e_recon_gl'].append(edge_recon_loss_gl_all/ len(train_dataloader))
            losses['node_kl'].append(node_kl_loss_all/ len(train_dataloader))
            losses['graph_kl'].append(graph_kl_loss_all/ len(train_dataloader))
            losses['e_recon_g'].append(edge_recon_loss_g_all/ len(train_dataloader))

            recon_current = edge_recon_loss_g_all/ len(train_dataloader) + edge_recon_loss_gl_all/ len(train_dataloader)


            if recon_current < best_recon_loss:
                best_recon_loss = recon_current
                torch.save(model.state_dict(), 'pretrained_models/tu_gvae_gcfx_{}_{}_seed_{}.pkl'.format(args.DS,model_save_name, seed))
                print('current best model saved!')



            print('Epoch {}, N_Recon GL {} E_Recon GL {} KL node {} KL graph {} E_Recon G {}'
                  .format(epoch, node_recon_loss_gl_all / len(train_dataloader),edge_recon_loss_gl_all / len(train_dataloader)
                          , node_kl_loss_all/ len(train_dataloader),graph_kl_loss_all/ len(train_dataloader), edge_recon_loss_g_all/ len(train_dataloader)))
            #scheduler.step(recon_current)


        #print('Saving the model')
        #torch.save(model.state_dict(), 'pretrained_models/mnist_gvae_disim_{}.pkl'.format(model_save_name))

        with open('loss_logs/losses_gvae_gcfx_tu_{}_{}_seed{}.json'.format(args.DS,model_save_name,seed), 'w') as f:
            json.dump(losses, f)