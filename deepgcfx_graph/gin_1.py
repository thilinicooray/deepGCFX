import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import Sequential, Linear, ReLU, Tanh, Sigmoid, PReLU, GRUCell, Embedding, Module, Parameter
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys

from summary_maker_1 import *

class FCNet(Module):
    """Simple class for non-linear fully connect network with gated tangent as in paper
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        in_dim = dims[0]
        out_dim = dims[1]
        self.first_lin = weight_norm(Linear(in_dim, out_dim), dim=None)
        self.tanh = Tanh()
        self.second_lin = weight_norm(Linear(in_dim, out_dim), dim=None)
        self.sigmoid = Sigmoid()


    def forward(self, x):

        y_hat = self.tanh(self.first_lin(x))
        g = self.sigmoid(self.second_lin(x))
        y = y_hat * g

        return y

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, summary_iter=2):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.eps = 1e-8
        self.dim = dim

        # self.nns = []
        #self.sum_q = Embedding(1, 128)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.act = PReLU()

        for i in range(num_gc_layers):

            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)


            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
        '''for i in range(num_gc_layers):

            if i == 0:
                conv = GCNConv(num_features, dim)
                bn = torch.nn.BatchNorm1d(dim)
            else:
                conv = GCNConv(dim, dim)
                bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)'''


        self.flatten = Linear(dim*num_gc_layers, dim)

        self.summary_maker = Summary_Maker(dim, iters=summary_iter)

        # node
        self.node_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_mu_bn = torch.nn.BatchNorm1d(dim)
        self.node_logvar_bn = torch.nn.BatchNorm1d(dim)

        self.graph_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_mu_bn = torch.nn.BatchNorm1d(dim)
        self.graph_logvar_bn = torch.nn.BatchNorm1d(dim)

    def forward(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

    def forward_tot4summary(self, x, edge_index, batch):


        xs = []
        tot = torch.zeros(x.size(0), self.dim).cuda()
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)
            tot += x


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker(tot, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


    def forward_tot4summary_diff(self, x, edge_index, batch):


        xs = []
        tot = torch.zeros(x.size(0), self.dim).cuda()
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)
            tot += x


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker(tot, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


    def forward_diff(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

    def forward_diff_weightedsum(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_dot_att(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

    def itercheck(self, x, edge_index, batch, iter, sample_idx):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_iter(x, batch, iter, sample_idx)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar



    def forward_weightedsum(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_dot_att(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar




    def forward_random(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_random(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar




    def forward_sum(self, x, edge_index, batch):


        xs = []
        x = F.relu(self.convs[0](x, edge_index))
        x = self.bns[0](x)
        for i in range(1,self.num_gc_layers-1):

            a = F.relu(self.convs[i](x, edge_index))
            x += self.bns[i](a)
            xs.append(x)

        slots, summary_related_node_info, noisy_node_info = self.summary_maker(x, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(summary_related_node_info)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(summary_related_node_info)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar, slots, summary_related_node_info, noisy_node_info


    def forward_softmax(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_softmax(x, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(summary_related_node_info)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(summary_related_node_info)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

class Encoder_simple(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, summary_iter=2):
        super(Encoder_simple, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.eps = 1e-8
        self.dim = dim

        # self.nns = []
        #self.sum_q = Embedding(1, 128)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.act = PReLU()

        for i in range(num_gc_layers):

            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)


            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
        '''for i in range(num_gc_layers):

            if i == 0:
                conv = GCNConv(num_features, dim)
                bn = torch.nn.BatchNorm1d(dim)
            else:
                conv = GCNConv(dim, dim)
                bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)'''


        self.flatten = Linear(dim*num_gc_layers, dim)

        self.summary_weighter = Linear(in_features=dim, out_features=1, bias=True)

        # node
        self.node_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_mu_bn = torch.nn.BatchNorm1d(dim)
        self.node_logvar_bn = torch.nn.BatchNorm1d(dim)

        self.graph_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_mu_bn = torch.nn.BatchNorm1d(dim)
        self.graph_logvar_bn = torch.nn.BatchNorm1d(dim)

    def forward(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        node_wise_common_weight = torch.sigmoid(self.summary_weighter(x))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*x
        noisy_node_info = (1-node_wise_common_weight)*x

        slots = global_add_pool(summary_related_node_info,batch)


        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

    def forward_tot4summary_diff(self, x, edge_index, batch):


        xs = []
        tot = torch.zeros(x.size(0), self.dim).cuda()
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)
            tot += x


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        #slots, summary_related_node_info, noisy_node_info = self.summary_maker(tot, batch)
        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        node_wise_common_weight = torch.sigmoid(self.summary_weighter(tot))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*tot
        noisy_node_info = (1-node_wise_common_weight)*tot

        slots = global_add_pool(summary_related_node_info,batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


    def forward_tot4summary_diff(self, x, edge_index, batch):


        xs = []
        tot = torch.zeros(x.size(0), self.dim).cuda()
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)
            tot += x


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x

        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        node_wise_common_weight = torch.sigmoid(self.summary_weighter(tot))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*tot
        noisy_node_info = (1-node_wise_common_weight)*tot

        slots = global_add_pool(summary_related_node_info,batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


    def forward_diff(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        node_wise_common_weight = torch.sigmoid(self.summary_weighter(x))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*x
        noisy_node_info = (1-node_wise_common_weight)*x

        slots = global_add_pool(summary_related_node_info,batch)
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


class Encoder_base(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, summary_iter=2):
        super(Encoder_base, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.eps = 1e-8

        # self.nns = []
        #self.sum_q = Embedding(1, 128)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.act = PReLU()

        for i in range(num_gc_layers):

            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)


            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
        '''for i in range(num_gc_layers):

            if i == 0:
                conv = GCNConv(num_features, dim)
                bn = torch.nn.BatchNorm1d(dim)
            else:
                conv = GCNConv(dim, dim)
                bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)'''


        self.flatten = Linear(dim*num_gc_layers, dim)


        # node
        self.node_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_mu_bn = torch.nn.BatchNorm1d(dim)
        self.node_logvar_bn = torch.nn.BatchNorm1d(dim)


    def forward(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(x)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(x)))


        return node_latent_space_mu,  node_latent_space_logvar




    def forward_sum(self, x, edge_index, batch):


        xs = []
        x = F.relu(self.convs[0](x, edge_index))
        x = self.bns[0](x)
        for i in range(1,self.num_gc_layers-1):

            a = F.relu(self.convs[i](x, edge_index))
            x += self.bns[i](a)
            xs.append(x)

        slots, summary_related_node_info, noisy_node_info = self.summary_maker(x, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(summary_related_node_info)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(summary_related_node_info)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar, slots, summary_related_node_info, noisy_node_info


    def forward_softmax(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        slots, summary_related_node_info, noisy_node_info = self.summary_maker.forward_softmax(x, batch)

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info)))
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info)))

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(summary_related_node_info)))
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(summary_related_node_info)))

        return node_latent_space_mu,  node_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar

class Decoder(torch.nn.Module):
    def __init__(self, node_dim, feat_size):
        super(Decoder, self).__init__()

        self.lin1 = torch.nn.Linear(in_features=node_dim, out_features=feat_size, bias=True)
        self.lin2 = torch.nn.Linear(in_features=node_dim, out_features=node_dim, bias=True)

        self.noderecon_bn = torch.nn.BatchNorm1d(feat_size)

    def forward(self, x):

        recon_node = self.lin1(x)
        recon_edge = x

        return recon_node, recon_edge




