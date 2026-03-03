import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param etypes: list of edge type strings
        """
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation, same as original
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        """
        :param G: HeteroData graph
        :param feat_dict: dict of node type -> feature tensor
        :return: dict of node type -> updated feature tensor
        """
        out_dict = {}

        for src_type, etype, dst_type in G.edge_types:
            if src_type not in feat_dict:
                continue

            edge_index = G[src_type, etype, dst_type].edge_index
            src_feat = feat_dict[src_type]

            # W_r * h for this relation
            Wh = self.weight[etype](src_feat)

            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # validate src index range
            assert src_idx.max().item() < Wh.size(0), \
                f"src_idx out of bounds for {src_type}: max={src_idx.max().item()}, size={Wh.size(0)}"

            # determine number of dst nodes using fixed graph size
            # must not use max(dst_idx)+1 because not every edge type covers all nodes
            # nodes with no incoming edges for this edge type will have agg=0
            if dst_type == 'target':
                num_dst = G['target'].x.size(0)
            elif dst_type in feat_dict:
                num_dst = feat_dict[dst_type].size(0)
            else:
                num_dst = G[dst_type].num_nodes

            # aggregate: mean of neighbor messages
            msg = Wh[src_idx]  # (num_edges, out_size)

            # accumulate sum
            agg = torch.zeros(num_dst, Wh.size(1), device=Wh.device)
            count = torch.zeros(num_dst, 1, device=Wh.device)
            agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(msg), msg)
            count.scatter_add_(0, dst_idx.unsqueeze(1), torch.ones(dst_idx.size(0), 1, device=Wh.device))
            count = count.clamp(min=1)
            agg = agg / count  # mean aggregation

            if dst_type not in out_dict:
                out_dict[dst_type] = agg
            else:
                out_dict[dst_type] = out_dict[dst_type] + agg

        return out_dict


class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        """
        :param ntype_dict: dict of node_type -> num_nodes for non-target nodes
        :param etypes: list of edge type strings
        :param in_size: input feature size for target node
        :param hidden_size: hidden layer size
        :param out_size: output size (number of classes)
        :param n_layers: number of hidden layers
        :param embedding_size: embedding size for non-target nodes
        """
        super(HeteroRGCN, self).__init__()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. A GPU is required to run this code.")
        self.device = torch.device('cuda')

        # trainable embeddings for non-target node types (featureless)
        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
            for ntype, num_nodes in ntype_dict.items() if ntype != 'target'
        }
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

        # layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))
        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, g, features):
        """
        :param g: HeteroData graph on GPU
        :param features: target node feature tensor on GPU
        :return: logits for target nodes
        """
        # build feature dict, same concept as original
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        h_dict['target'] = features

        # pass through all RGCN layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # output layer on target node only
        return self.layers[-1](h_dict['target'])
