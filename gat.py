import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import config

import re

import networkx as nx   
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

import ipdb

from get_role import get_role

def create_graph(text, character, embeddings):

    num_nodes = len(text)
    roles = get_role()

    x = embeddings
    edge_index = None
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # ipdb.set_trace()
                exist_roles1 = [role for role in roles if re.search(role, text[i])]
                exist_roles2 = [role for role in roles if re.search(role, text[j])]
                if set(exist_roles1).intersection(set(exist_roles2)):  # 交集不为空
                    # ipdb.set_trace()
                    if edge_index is None:
                        edge_index = torch.tensor([[i, j]])
                    else:
                        # 无向图
                        edge_index = torch.cat([edge_index, torch.tensor([[i, j]])], dim=0)
                        edge_index = torch.cat([edge_index, torch.tensor([[j, i]])], dim=0)
            # else:  # 自回路 考虑吗？
            #     if edge_index is None:
            #         edge_index = torch.tensor([[i, i]])
            #     else:
            #         edge_index = torch.cat([edge_index, torch.tensor([[i, i]])], dim=0)

    
    try:
        edge_index = edge_index.t().contiguous()  # 考虑没有边的情况
        #ipdb.set_trace()
    except:
        #ipdb.set_trace()
        print(len(text))
        print(text)
        edge_index = torch.tensor([[], []], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.to(config.device))

    return data


def draw_graph(edge_index, name=None):
    G = nx.Graph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        G.add_edge(i, j)
    plt.figure(figsize=(8, 8)) # 设置画布的大小
    nx.draw_networkx(G)
    plt.show()
    # plt.savefig('{}.png'.format(name if name else 'path'))


def draw_graph_2(Data):
    G = to_networkx(Data)
    nx.draw(G)
    plt.savefig("path.png")
    plt.show()

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()

        # num_features: Alias for num_node_features.
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.1)
        self.conv1_1 = GATConv(8*8, 8, heads=8, dropout=0.1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.2)

    def forward(self, x, edge_index):
        # ipdb.set_trace()
        x_copy = x.clone()
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1_1(x, edge_index))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x + x_copy  # Residual connection, 避免孤立节点变成全0
        
        # return F.log_softmax(x, dim=-1)  # log_softmax ??
        return x   #  我觉得这个位置还不要softmax


# y = torch.tensor([1, 4, 5], dtype=torch.int64)
# train_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
# data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)



if __name__ == '__main__':
    model = GAT(3, 6)

    x = torch.tensor([[1.2,2.2,3.1], [0,1.1,1], [1,2.5,3]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # draw_graph(data.edge_index)
    draw_graph_2(data)
    
    print(data)
    pred = model(data.x, data.edge_index)
    print(pred.shape)  # [torch.FloatTensor of size (3, 6)]

