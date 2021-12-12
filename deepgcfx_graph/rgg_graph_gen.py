import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch_geometric.utils import erdos_renyi_graph, from_networkx
import numpy as np
import random
import os
import os.path as osp
from torch_geometric.data import DataLoader
import networkx as nx

class SyntheticRGGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SyntheticRGGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        data_list = []
        n = 50
        radius = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8, 0.9]
        labels = [0,1,2,3,4,5,6,7,8]

        for i in range(10000):

            idx = np.random.randint(0,len(radius)-1)

            r = radius[idx]

            RGG = nx.random_geometric_graph(n, radius=r)
            data = from_networkx(RGG)
            node_attrs = list(next(iter(RGG.nodes(data=True)))[-1].keys())
            group_node_attrs = list(node_attrs)
            xs = [data[key] for key in group_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
            data.y = labels[idx]

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':

    seed = 123

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SyntheticRGG')

    dataset = SyntheticRGGDataset(path).shuffle()

    train_dataset = dataset[:8000]
    test_dataset = dataset[8000:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=2)

    for data in train_loader:
        data = data.to(device)
        print('data ', data.batch, data.x, data.y)
        break



