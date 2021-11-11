import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

from get_role import get_role
import re

import ipdb

def create_graph(text, character, embeddings):

    num_nodes = len(text)
    roles = get_role()

    x = embeddings
    edge_index = None
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
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

                

    x = embeddings
    edge_index = edge_index.t().contiguous()
    data = Data(x=x, edge_index=edge_index)

    return data


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()

        # num_features: Alias for num_node_features.
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)

        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)  # log_softmax ??


# y = torch.tensor([1, 4, 5], dtype=torch.int64)
# train_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
# data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


if __name__ == '__main__':
    model = GAT(3, 6)

    x = torch.tensor([[1.2,2.2,3.1], [0,1.1,1], [1,2.5,3]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    print(data)
    pred = model(data.x, data.edge_index)
    print(pred.shape)  # [torch.FloatTensor of size (3, 6)]

