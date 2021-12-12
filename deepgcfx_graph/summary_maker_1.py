import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool
from numpy import savetxt
from scipy import stats

class Summary_Maker(nn.Module):
    def __init__(self, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super(Summary_Maker, self).__init__()
        self.num_slots = 1
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn( 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros( 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.g_weight_calc = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1))

        #self.init_weight_calc = nn.Linear(in_features=dim, out_features=1, bias=True)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.mlp_nodes = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_nodes  = nn.LayerNorm(dim)

    def forward(self, inputs, batch):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        '''node_wise_common_weight = torch.sigmoid(self.init_weight_calc(inputs))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*inputs
        noisy_node_info = (1-node_wise_common_weight)*inputs

        slots = global_add_pool(summary_related_node_info,batch)'''

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None

        for iter in range(self.iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            compact_node = torch.sigmoid(k)
            mask = (compact_node >= node_wise_summary_weight).float()
            antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = mask * v
            noisy_node_info = antimask * v

            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)

        slots_a = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots_a)
        q = self.to_q(slots)

        expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

        compact_node = torch.sigmoid(k)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()

        summary_related_node_info = mask * v
        noisy_node_info = antimask * v

        return slots, summary_related_node_info, noisy_node_info

    def forward_iter(self, inputs, batch, iters, sample_idx):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        '''node_wise_common_weight = torch.sigmoid(self.init_weight_calc(inputs))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*inputs
        noisy_node_info = (1-node_wise_common_weight)*inputs

        slots = global_add_pool(summary_related_node_info,batch)'''

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None



        for iter in range(iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            compact_node = torch.sigmoid(k)
            mask = (compact_node >= node_wise_summary_weight).float()
            antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = mask * v
            noisy_node_info = antimask * v

            graph_embedding_expanded = torch.repeat_interleave(slots, count, dim=0)

            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)



        slots_a = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots_a)
        q = self.to_q(slots)

        expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

        compact_node = torch.sigmoid(k)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()

        summary_related_node_info = mask * v
        noisy_node_info = antimask * v

        #n_rho, n_pval = stats.spearmanr(summary_related_node_info[0].squeeze().cpu().numpy(),noisy_node_info[0].squeeze().cpu().numpy())

        #print('uncommon to g after iter {}'.format(iters), n_rho )

        return slots, summary_related_node_info, noisy_node_info

    def forward_random(self, inputs, batch):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        slots_a = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots_a)
        q = self.to_q(slots)

        expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

        compact_node = torch.sigmoid(k)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()

        summary_related_node_info = mask * v
        noisy_node_info = antimask * v

        return slots, summary_related_node_info, noisy_node_info

    def forward_dot_att(self, inputs, batch):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1
        node_g_align = F.one_hot(batch)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        '''node_wise_common_weight = torch.sigmoid(self.init_weight_calc(inputs))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*inputs
        noisy_node_info = (1-node_wise_common_weight)*inputs

        slots = global_add_pool(summary_related_node_info,batch)'''

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None

        for iter in range(self.iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

            #compact_node = torch.sigmoid(v)
            #mask = (compact_node >= node_wise_summary_weight).float()
            #antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = node_wise_summary_weight * v
            #noisy_node_info = antimask * v

            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)

        slots = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))
        node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

        '''compact_node = torch.sigmoid(v)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()'''

        summary_related_node_info = node_wise_summary_weight * v
        noisy_node_info = (1-node_wise_summary_weight) * v

        return slots, summary_related_node_info, noisy_node_info

    def forward_dot_att_iter(self, inputs, batch, iters):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1
        node_g_align = F.one_hot(batch)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        '''node_wise_common_weight = torch.sigmoid(self.init_weight_calc(inputs))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*inputs
        noisy_node_info = (1-node_wise_common_weight)*inputs

        slots = global_add_pool(summary_related_node_info,batch)'''

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None

        init_slot = slots

        for iter in range(iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

            #compact_node = torch.sigmoid(v)
            #mask = (compact_node >= node_wise_summary_weight).float()
            #antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = node_wise_summary_weight * v
            noisy_node_info = (1-node_wise_summary_weight) * v

            #print('q size', q.size())

            graph_embedding_expanded = torch.repeat_interleave(q, count, dim=0)

            #print('val ', graph_embedding_expanded.size(), k.size())

            #n_rho, n_pval = stats.spearmanr(torch.cat([graph_embedding_expanded,k],0) .cpu().numpy(), axis=1)
            #savetxt('accum_analyze/originalnode2common_rho_{}.csv'.format(iter), n_rho, delimiter=',')

            n_rho, n_pval = stats.spearmanr(slots.squeeze().cpu().numpy(),summary_related_node_info[0].squeeze().cpu().numpy())

            print('summary to corre after iter {}'.format(iters), n_rho )

            n_rho, n_pval = stats.spearmanr(slots.squeeze().cpu().numpy(),noisy_node_info[0].squeeze().cpu().numpy())

            print('noisy to corre after iter {}'.format(iters), n_rho )




            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)
            #get embedding


            #savetxt('accum_analyze/g_iter_accuminside{}.csv'.format(iter), summary_related_node_info.cpu().numpy(), delimiter=',')
            #savetxt('accum_analyze/n_iter_accuminside{}.csv'.format(iter), noisy_node_info.cpu().numpy(), delimiter=',')

            #n_rho, n_pval = stats.spearmanr(torch.cat([summary_related_node_info,noisy_node_info],0) .cpu().numpy(), axis=1)
            #savetxt('accum_analyze/uncommoncommon_rho_{}.csv'.format(iter), n_rho, delimiter=',')

        #print('shapes of slots ', init_slot.size(), slots.size())
        #check correlation

        n_rho, n_pval = stats.spearmanr(init_slot.squeeze().cpu().numpy(),slots.squeeze().cpu().numpy())

        print('corre after iter {}'.format(iters), n_rho )

        slots = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))
        node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

        '''compact_node = torch.sigmoid(v)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()'''

        summary_related_node_info = node_wise_summary_weight * v
        noisy_node_info = (1-node_wise_summary_weight) * v

        return slots, summary_related_node_info, noisy_node_info

    def forward_dot(self, inputs, batch):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1
        node_g_align = F.one_hot(batch)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None

        for iter in range(self.iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

            compact_node = torch.sigmoid(k)
            mask = (compact_node >= node_wise_summary_weight).float()
            antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = mask * v
            #noisy_node_info = antimask * v

            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)

        slots = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        #expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        #node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))
        node_wise_summary_weight = torch.sum(torch.sigmoid(torch.mm(k, q.t())) * node_g_align,-1).unsqueeze(-1)

        compact_node = torch.sigmoid(k)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()

        summary_related_node_info = mask * v
        noisy_node_info = antimask * v

        return slots, summary_related_node_info, noisy_node_info





