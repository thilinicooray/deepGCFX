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

from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing

import torch_geometric
from torch_geometric.datasets import CitationFull,WebKB,WikipediaNetwork,Actor,Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
from torch.nn import Sequential, Linear, ReLU, Tanh, Sigmoid, PReLU
import sys
import json
from torch import optim

from torch_geometric.nn import GINConv, global_add_pool, GCNConv

from sklearn.linear_model import LogisticRegression
from kmeans_pytorch import kmeans
#import faiss

from args import arg_parse
from utils import *
from gcn_1 import *

import torch
import torch.nn as nn

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        X = X.cpu().numpy()
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.cpu().numpy(), 1)[1]


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
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers):
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

        graph_kl_divergence_loss = (-0.5 ) * torch.mean(torch.sum(
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


        edge_recon_loss_gl = self.recon_loss(full_node_feat_gl, edge_index, batch)
        edge_recon_loss_g = self.recon_loss(gnode_latent_embeddings_unnorm, edge_index, batch)

        '''contrastive_loss = self.node_graph_contrastive_loss(self.g_proj(graph_latent_embeddings), self.n_proj(gnode_latent_embeddings_unnorm),
                                                            self.n_proj(node_latent_embeddings_unnorm), batch)'''



        #loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl + contrastive_loss + edge_recon_loss_g
        loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl  + edge_recon_loss_g

        loss.backward()

        return  node_kl_divergence_loss.item(), edge_recon_loss_gl.item()


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

        rec = torch.sigmoid(torch.bmm(a, a_t))

        org_adj = to_dense_adj(edge_index, batch)


        #pos_weight = float(z.size(0) * z.size(0) - org_adj.sum()) / org_adj.sum()
        #norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - org_adj.sum()) * 2)

        #loss = norm * F.binary_cross_entropy_with_logits(rec, org_adj, pos_weight=pos_weight)
        loss = F.binary_cross_entropy_with_logits(rec, org_adj)
        return loss

    def get_embedding(self, dataset, train_mask, val_mask, test_mask):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():

            dataset.to(device)
            x, edge_index = dataset.x, dataset.edge_index


            node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(x, edge_index, torch.zeros(dataset.x.size(0)).long().to(device))

            node_latent = reparameterize(training=False, mu=node_mu, logvar=node_logvar)
            graph_latent_embeddings = reparameterize(training=False, mu=graph_mu, logvar=graph_logvar)

            _, count = torch.unique(torch.zeros(dataset.x.size(0)).long().to(device),  return_counts=True)

            graph_embedding_expanded = torch.repeat_interleave(graph_latent_embeddings, count, dim=0)

            fused = self.fusioner(torch.cat([graph_embedding_expanded,node_latent], -1))

        train_emb = node_latent[train_mask]
        train_y = dataset.y[train_mask]
        val_emb = node_latent[val_mask]
        val_y = dataset.y[val_mask]
        test_emb = node_latent[test_mask]
        test_y = dataset.y[test_mask]
        train_emb_c = graph_embedding_expanded[train_mask]
        val_emb_c = graph_embedding_expanded[val_mask]
        test_emb_c = graph_embedding_expanded[test_mask]
        train_emb_fused = fused[train_mask]
        val_emb_fused = fused[val_mask]
        test_emb_fused = fused[test_mask]


        return train_emb,train_y,train_emb_c, train_emb_fused,val_emb,val_y,val_emb_c, val_emb_fused, \
               test_emb,test_y,test_emb_c, test_emb_fused


    def get_embedding_full(self, dataset):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():

            dataset.to(device)
            x, edge_index = dataset.x, dataset.edge_index


            node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(x, edge_index, torch.zeros(dataset.x.size(0)).long().to(device))

            node_latent = reparameterize(training=False, mu=node_mu, logvar=node_logvar)
            graph_latent_embeddings = reparameterize(training=False, mu=graph_mu, logvar=graph_logvar)

            _, count = torch.unique(torch.zeros(dataset.x.size(0)).long().to(device),  return_counts=True)

            graph_embedding_expanded = torch.repeat_interleave(graph_latent_embeddings, count, dim=0)

            fused = self.fusioner(torch.cat([graph_embedding_expanded,node_latent], -1))


        return node_latent, dataset.y, graph_embedding_expanded, fused


def test(train_z, train_y, val_z, val_y,test_z, test_y,  solver='lbfgs',
         multi_class='ovr', *args, **kwargs):
    r"""Evaluates latent space quality via a logistic regression downstream
    task."""

    log_reg = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=15000)
    clf = MultiOutputClassifier(log_reg)

    scaler = preprocessing.StandardScaler().fit(train_z)

    updated = scaler.transform(train_z)


    clf.fit(updated,train_y)

    predict_val = clf.predict(scaler.transform(val_z))

    micro_f1_val = f1_score(val_y, predict_val, average='micro')

    predict_test = clf.predict(scaler.transform(test_z))

    micro_f1_test = f1_score(test_y, predict_test, average='micro')

    return micro_f1_val, micro_f1_test

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            #nn.Linear(in_dim, hid_dim),
            #nn.ReLU(),
            #nn.Dropout(dropout, inplace=True),
            nn.Linear(in_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        logits = self.main(x)
        return logits

def get_masks(dataset, current_train_split, current_val_split):

    classes, counts = torch.unique(dataset.y, return_counts=True)

    print('Num classifying classes ', classes.size(), classes)

    train_mask = torch.zeros(dataset.y.size(0)).long().cuda() > 0
    val_mask = torch.zeros(dataset.y.size(0)).long().cuda() > 0
    test_mask = torch.zeros(dataset.y.size(0)).long().cuda() > 0

    for c in classes:
        num_train_per_class = torch.round(counts[c]*0.6).long()
        idx = (dataset.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)


    remaining = remaining[torch.randperm(remaining.size(0))]

    val_mask[remaining[:current_val_split]] = True

    test_mask[remaining[current_val_split:]] = True

    #print('final mask sizes', current_train_split, current_val_split,torch.sum(train_mask), torch.sum(val_mask), torch.sum(test_mask))

    return train_mask, val_mask, test_mask


def model_training(dataset,dataset_num_features, hidden_dim, num_layers, max_epoch, ds, model_name):

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = deepGCFX(dataset_num_features,hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    model.train()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience=20
    print('deepGCFX training ....')
    for epoch in range(1, max_epoch+1):

        optimizer.zero_grad()
        node_kl_divergence_loss, edge_recon_loss_gl = model(dataset.x, dataset.edge_index, torch.zeros(dataset.x.size(0)).long().to(device),0)

        #loss = node_kl_divergence_loss + edge_recon_loss_gl

        #print('Recon Loss:', epoch, edge_recon_loss_gl)

        optimizer.step()

        if edge_recon_loss_gl < best:
            best = edge_recon_loss_gl
            best_t = epoch
            cnt_wait = 0
            #print('best model saved!')
            torch.save(model.state_dict(), 'best_gcfx_class_{}_{}.pkl'.format(ds,model_name))
        else:
            cnt_wait += 1

        '''if cnt_wait == patience:
            print('Early stopping!')
            break'''

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_gcfx_class_{}_{}.pkl'.format(ds,model_name)))

    return model

def clustering(emb,y):

    print('k-means clustering started!')


    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score

    full_ari = []
    full_nmi = []

    for i in range(1):

        #kmeans = KMeans(n_clusters=n_classes).fit_predict(emb)
        kmeans = FaissKMeans(n_clusters=n_classes)
        kmeans.fit(emb)
        cluster_ids_x = kmeans.predict(emb)
        cluster_ids_x = np.squeeze(cluster_ids_x)

        '''cluster_ids_x, cluster_centers = kmeans(
            X=emb, num_clusters=n_classes, distance='euclidean', device=torch.device('cuda')
        )'''
        ARI = adjusted_rand_score(y.cpu().numpy(), cluster_ids_x)
        NMI = normalized_mutual_info_score(y.cpu().numpy(), cluster_ids_x)

        full_ari.append(ARI)
        full_nmi.append(NMI)

    #print('AVG NMI ARI Non-common Only', np.mean(np.array(full_nmi)), np.std(np.array(full_nmi)), np.mean(np.array(full_ari)),np.std(np.array(full_ari)),)

    return np.mean(np.array(full_nmi)), np.mean(np.array(full_ari))

def classification(hidden_dim, train_emb, train_y_labels, val_emb, val_y_labels, test_emb, test_y_labels):

    print('Logistic regression started!')

    xent = nn.CrossEntropyLoss()


    perf4meantest = []
    perf4meanval = []
    for _ in range(1):
        log = LogReg(hidden_dim, n_classes).cuda()
        opt = torch.optim.Adam(log.parameters(), lr=1e-2)
        log.cuda()

        acc_val_best = 0
        acc_test_best = 0

        for iter in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_emb)
            loss = xent(logits, train_y_labels)

            loss.backward()
            opt.step()

            log.eval()

            logits_test = log(test_emb)
            preds_test = torch.argmax(logits_test, dim=1)
            acc_test = torch.sum(preds_test == test_y_labels).float() / test_y_labels.shape[0]

            logits_val = log(val_emb)
            preds_val = torch.argmax(logits_val, dim=1)
            acc_val = torch.sum(preds_val == val_y_labels).float() / val_y_labels.shape[0]


            if acc_val > acc_val_best:
                acc_val_best = acc_val
                acc_test_best = acc_test
        perf4meantest.append(acc_test_best)
        perf4meanval.append(acc_val_best)

    mean_best_val = torch.as_tensor(perf4meanval).mean()
    mean_best_test = torch.as_tensor(perf4meantest).mean()

    return mean_best_val, mean_best_test


def get_results(seed, best_model,dataset, hidden_dim, num_layers, max_epoch, ds, model_name,current_train_split, current_val_split, runs, gate_val):

    dataset_num_features = dataset.x.size(-1)

    best_model.eval()

    #get all types of embeddings for clustering
    non_common_only_all, y_all, common_only_all, fused_all = best_model.get_embedding_full(dataset)

    # get masks for splits
    #train_mask, val_mask, test_mask = get_masks(dataset, current_train_split, current_val_split)
    train_mask, val_mask, test_mask = dataset.train_mask[:,seed],dataset.val_mask[:,seed],dataset.test_mask[:,seed]
    non_common_only_train, y_train, common_only_train, fused_train,non_common_only_val, y_val, common_only_val, \
    fused_val, non_common_only_test, y_test, common_only_test, fused_test \
        = best_model.get_embedding(dataset, train_mask, val_mask, test_mask)

    #1. non_common only
    print('evaluating non-common only')
    #nc_nmi, nc_ari = clustering(non_common_only_all,y_all)
    nc_val_acc, nc_test_acc = classification(hidden_dim,torch.tanh(non_common_only_train),y_train,torch.tanh(non_common_only_val),y_val,
                                             torch.tanh(non_common_only_test),y_test)

    #2. fused
    print('evaluating fused')
    #fused_nmi, fused_ari = clustering(fused_all,y_all)
    fused_all_val_acc, fused_all_test_acc = classification(hidden_dim,fused_train,y_train,fused_val,y_val,
                                                           fused_test,y_test)

    #3. weighted c, nc
    print('evaluating weighted')
    weighted = {}

    for coef in range(runs+1):
        lamda = coef * gate_val
        lamda = round(lamda,2)
        #print('current lamda ', lamda)

        '''rep_all = lamda * non_common_only_all + (1 - lamda)*common_only_all
        rep_train = lamda * non_common_only_train + (1 - lamda)*common_only_train
        rep_val = lamda * non_common_only_val + (1 - lamda)*common_only_val
        rep_test = lamda * non_common_only_test + (1 - lamda)*common_only_test'''

        '''rep_all = torch.cat([lamda * non_common_only_all, (1 - lamda)*common_only_all],-1)
        rep_train = torch.cat([lamda * non_common_only_train, (1 - lamda)*common_only_train],-1)
        rep_val = torch.cat([lamda * non_common_only_val , (1 - lamda)*common_only_val],-1)
        rep_test = torch.cat([lamda * non_common_only_test , (1 - lamda)*common_only_test],-1)'''

        '''rep_all = lamda * non_common_only_all * (1 - lamda)*common_only_all
        rep_train = lamda * non_common_only_train * (1 - lamda)*common_only_train
        rep_val = lamda * non_common_only_val * (1 - lamda)*common_only_val
        rep_test = lamda * non_common_only_test * (1 - lamda)*common_only_test'''


        rep_all = torch.exp(lamda * non_common_only_all + (1 - lamda)*common_only_all)
        rep_train = lamda * non_common_only_train + (1 - lamda)*common_only_train
        rep_val = lamda * non_common_only_val + (1 - lamda)*common_only_val
        rep_test = lamda * non_common_only_test + (1 - lamda)*common_only_test


        #w_nmi, w_ari = clustering(rep_all,y_all)
        w_val_acc, w_test_acc = classification(hidden_dim,rep_train,y_train,rep_val,y_val,
                                               rep_test,y_test)
        #weighted[coef] = (lamda,w_nmi,w_ari,w_val_acc,w_test_acc)
        weighted[coef] = (lamda,None,None,w_val_acc,w_test_acc)


    #return {'nc_only':(nc_nmi, nc_ari,nc_test_acc),'fused':(fused_nmi, fused_ari, fused_all_test_acc),'weighted':weighted}
    return {'nc_only':(None, None,nc_test_acc),'fused':(None, None, fused_all_test_acc),'weighted':weighted}



def all_seed_perf(dataset, hidden_dim, num_layers, max_epoch, ds, model_name,current_train_split, current_val_split):

    perf_nc_only_nmi = []
    perf_nc_only_ari = []
    perf_nc_only_acc = []
    perf_fused_nmi = []
    perf_fused_ari = []
    perf_fused_acc = []
    perf_weighted_nmi = {}
    perf_weighted_ari = {}
    perf_weighted_acc_val = {}
    perf_weighted_acc_test = {}
    runs = 20
    gate_val = 0.05
    #for seed in [0,1,2,3,4,5,6,7,8,9]:
    dataset_num_features = dataset.x.size(-1)
    best_model = model_training(dataset,dataset_num_features, hidden_dim, num_layers, max_epoch, ds, model_name)
    for seed in range(10):
        print('seed ', seed, 'epochs ', max_epoch)

        #in case dataset doesnt provide the random split, this seed is used


        result_dict = get_results(seed,best_model,dataset, hidden_dim, num_layers, max_epoch, ds, model_name,current_train_split, current_val_split, runs, gate_val)

        perf_nc_only_nmi.append(result_dict['nc_only'][0])
        perf_nc_only_ari.append(result_dict['nc_only'][1])
        perf_nc_only_acc.append(result_dict['nc_only'][2])
        perf_fused_nmi.append(result_dict['fused'][0])
        perf_fused_ari.append(result_dict['fused'][1])
        perf_fused_acc.append(result_dict['fused'][2])

        for coef in range(runs+1):
            if coef not in perf_weighted_nmi:
                perf_weighted_nmi[coef] = [result_dict['weighted'][coef][1]]
                perf_weighted_ari[coef] = [result_dict['weighted'][coef][2]]
                perf_weighted_acc_val[coef] = [result_dict['weighted'][coef][3]]
                perf_weighted_acc_test[coef] = [result_dict['weighted'][coef][4]]
            else:
                perf_weighted_nmi[coef].append(result_dict['weighted'][coef][1])
                perf_weighted_ari[coef].append(result_dict['weighted'][coef][2])
                perf_weighted_acc_val[coef].append(result_dict['weighted'][coef][3])
                perf_weighted_acc_test[coef].append(result_dict['weighted'][coef][4])

    #print('perf_nc_only_nmi ', perf_nc_only_nmi)
    #all_seed_nc_only_nmi = torch.as_tensor(perf_nc_only_nmi)
    #all_seed_nc_only_ari = torch.as_tensor(perf_nc_only_ari)
    all_seed_nc_only_acc = torch.as_tensor(perf_nc_only_acc)
    #all_seed_fused_nmi = torch.as_tensor(perf_fused_nmi)
    #all_seed_fused_ari = torch.as_tensor(perf_fused_ari)
    all_seed_fused_acc = torch.as_tensor(perf_fused_acc)

    #print('{} nc only nmi mean std :'.format(ds), all_seed_nc_only_nmi.mean().item(), all_seed_nc_only_nmi.std().item())
    #print('{} nc only ari mean std :'.format(ds), all_seed_nc_only_ari.mean().item(), all_seed_nc_only_ari.std().item())
    #print('{} nc only test acc mean std :'.format(ds), all_seed_nc_only_acc.mean().item(), all_seed_nc_only_acc.std().item())
    #print('{} fused nmi mean std :'.format(ds), all_seed_fused_nmi.mean().item(), all_seed_fused_nmi.std().item())
    #print('{} fused ari mean std :'.format(ds), all_seed_fused_ari.mean().item(), all_seed_fused_ari.std().item())
    print('{} fused test acc mean std :'.format(ds), all_seed_fused_acc.mean().item(), all_seed_fused_acc.std().item())

    weighted_val = {}
    weighted_test = {}
    weighted_nmi = {}
    weighted_ari = {}

    for coef in range(runs+1):
        lamda = coef * gate_val
        lamda = round(lamda,2)

        #lamda = weight of nc
        #all_seed_w_nmi = torch.as_tensor(perf_weighted_nmi[coef])
        #all_seed_w_ari = torch.as_tensor(perf_weighted_ari[coef])
        all_seed_w_acc_val = torch.as_tensor(perf_weighted_acc_val[coef])
        all_seed_w_acc_test = torch.as_tensor(perf_weighted_acc_test[coef])

        weighted_val[coef] = all_seed_w_acc_val.mean()
        weighted_test[coef] = (all_seed_w_acc_test.mean(),all_seed_w_acc_test.std())
        #weighted_nmi[coef] = all_seed_w_nmi.mean()
        #weighted_ari[coef] = all_seed_w_ari.mean()

        '''print('\n')
        print('{} lambda nmi mean std :'.format(lamda), all_seed_w_nmi.mean().item(), all_seed_w_nmi.std().item())
        print('{} lambda ari mean std :'.format(lamda), all_seed_w_ari.mean().item(), all_seed_w_ari.std().item())
        print('{} lambda val acc mean std :'.format(lamda), all_seed_w_acc_val.mean().item(), all_seed_w_acc_val.std().item())
        print('{} lambda test acc mean std :'.format(lamda), all_seed_w_acc_test.mean().item(), all_seed_w_acc_test.std().item())
        print('\n')'''

    #sorted_nmi = sorted(weighted_nmi.items(), reverse=True, key=lambda kv: kv[1])

    #sorted_ari = sorted(weighted_ari.items(), reverse=True, key=lambda kv: kv[1])

    sorted_val = sorted(weighted_val.items(), reverse=True, key=lambda kv: kv[1])

    '''print('{} best weighted performance :'.format(ds),'\n\t','nmi :', sorted_nmi[0][0]*gate_val, sorted_nmi[0][1],
          '\n\t','ari :', sorted_ari[0][0]*gate_val, sorted_ari[0][1],
          '\n\t','val :', sorted_val[0][0]*gate_val, sorted_val[0][1],
          '\n\t','test :', weighted_test[sorted_val[0][0]])'''
    print('{} best weighted performance :'.format(ds),'\n\t',
          '\n\t','val :', sorted_val[0][0]*gate_val, sorted_val[0][1],
          '\n\t','test :', weighted_test[sorted_val[0][0]])
    print('{} nc only performance :'.format(ds),'\n\t',
          '\n\t','test :', weighted_test[runs])
    print('{} c only performance :'.format(ds),'\n\t',
          '\n\t','test :', weighted_test[0])



if __name__ == '__main__':

    args = arg_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name



    '''#citation dataset

    #split_idx = {'Cora':{'train':11876,'val':3958}, 'CiteSeer':{'train':2538,'val':846},'PubMed':{'train':11830,'val':3943}}#citation full
    split_idx = {'Cora':{'train':1625,'val':541}, 'CiteSeer':{'train':1996,'val':665},'PubMed':{'train':11830,'val':3943}}#planetoid

    #for ds in ["Cora", "CiteSeer", "PubMed"]:
    for ds in ["PubMed"]:

        print(' current dataset:', ds)

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', ds)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        #dataset = CitationFull(path, name=ds)[0].cuda()
        dataset = Planetoid(path, name=ds)[0].cuda()

        print('dataset stats ', dataset)

        current_train_split = split_idx[ds]['train']
        current_val_split = split_idx[ds]['val']

        classes, _ = torch.unique(dataset.y, return_counts=True)
        n_classes = classes.size(0)

        max_epoch = args.num_epochs
        num_gc_layers = args.num_gc_layers
        hidden_dim = args.hidden_dim
        all_seed_perf(dataset, hidden_dim, num_gc_layers, max_epoch, ds, model_name,current_train_split, current_val_split)'''




    '''#webkb dataset

    split_idx = {'Cornell':{'train':110,'val':36}, 'Texas':{'train':110,'val':36},'Wisconsin':{'train':151,'val':50}}

    #for ds in ["Cornell", "Texas", "Wisconsin"]:
    for ds in ["Cornell"]:

        print(' current dataset:', ds)

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', ds)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        dataset = WebKB(path, name=ds)[0].cuda()

        print('dataset stats ', dataset)

        current_train_split = split_idx[ds]['train']
        current_val_split = split_idx[ds]['val']

        classes, _ = torch.unique(dataset.y, return_counts=True)
        n_classes = classes.size(0)

        max_epoch = args.num_epochs
        num_gc_layers = args.num_gc_layers
        hidden_dim = args.hidden_dim

        all_seed_perf(dataset, hidden_dim, num_gc_layers, max_epoch, ds, model_name,current_train_split, current_val_split)'''








    '''#wikipedia network

    split_idx = {'chameleon':{'train':1366,'val':455}, 'squirrel':{'train':3121,'val':1040}}

    #for ds in ["chameleon", "squirrel"]:
    for ds in ["squirrel"]:

        print(' current dataset:', ds)

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', ds)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        dataset = WikipediaNetwork(path, name=ds, geom_gcn_preprocess=True)[0].cuda()

        print('dataset stats ', dataset)

        current_train_split = split_idx[ds]['train']
        current_val_split = split_idx[ds]['val']

        classes, _ = torch.unique(dataset.y, return_counts=True)
        n_classes = classes.size(0)

        if ds == 'chameleon':
            max_epoch = 2000
            num_gc_layers = 1
            hidden_dim = 512
            #recon1
            #decay 5e-5

        max_epoch = args.num_epochs
        num_gc_layers = args.num_gc_layers
        hidden_dim = args.hidden_dim

        all_seed_perf(dataset, hidden_dim, num_gc_layers, max_epoch, ds, model_name,current_train_split, current_val_split)'''




    #actor network

    split_idx = {'train':4560,'val':1520}

    ds = "Actor"
    print(' current dataset:', ds)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', ds)

    dataset = Actor(path)[0].cuda()

    print('dataset stats ', dataset)

    current_train_split = split_idx['train']
    current_val_split = split_idx['val']

    classes, _ = torch.unique(dataset.y, return_counts=True)
    n_classes = classes.size(0)

    max_epoch = 2000
    num_gc_layers = 1
    hidden_dim = 512

    all_seed_perf(dataset, hidden_dim, num_gc_layers, max_epoch, ds, model_name,current_train_split, current_val_split)





