import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import svd_lowrank
from mask import*
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import negative_sampling, degree
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import calculate_metrics
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from torch_geometric.utils import negative_sampling
from torch.nn import Module, Linear, Sigmoid
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
class MP(MessagePassing):
    def __init__(self):
        super(MP, self).__init__()

    def message(self, x_j, norm=None):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp.data.tolist())


class BasisGenerator(nn.Module):
    """
    generate all the feature spaces
    """
    def __init__(self, nx, nlx, nl, K, poly, low_x=False, low_lx=False, low_l=True, norm1=False):
        super(BasisGenerator, self).__init__()
        self.nx = nx
        self.nlx = nlx
        self.nl = nl
        self.norm1 = norm1
        self.K = K  # for lx
        self.poly = poly  # for lx
        self.low_x = low_x
        self.low_lx = low_lx
        self.low_l = low_l
        self.mp = MP()

    def get_x_basis(self, x):
        x = F.normalize(x, dim=1)
        x = F.normalize(x, dim=0)
        if self.low_x:
            U, S, V = svd_lowrank(x, q=self.nx)
            low_x = torch.mm(U, torch.diag(S))
            return low_x
        else:
            return x

    def get_lx_basis(self, x, edge_index):
        # generate all feature spaces
        lxs = []
        num_nodes = x.shape[0]
        lap_edges, lap_norm = get_laplacian(edge_index=edge_index,
                                            normalization='sym',
                                            num_nodes=num_nodes)  # 标准的归一化后lap
        h = F.normalize(x, dim=1)

        if self.poly == 'gcn':
            lxs = [h]
            #
            edges, norm = add_self_loops(edge_index=lap_edges,
                                         # edge_weight=-lap_norm,
                                         fill_value=2.,
                                         num_nodes=num_nodes)  # \hat{A} = I + \tilde{A}
            edges, norm = get_laplacian(edge_index=edges,
                                        # edge_weight=norm,
                                        normalization='sym',
                                        num_nodes=num_nodes)  # \hat{L}
            edges, norm = add_self_loops(edge_index=edges,
                                         # edge_weight=-norm,
                                         fill_value=1.,
                                         num_nodes=num_nodes)
            # may use directly gcn-norm
            # gcn_norm(edge_index=edge_index, num_nodes=num_nodes)

            for k in range(self.K + 1):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'gpr':
            lxs = [h]
            edges, norm = add_self_loops(edge_index=lap_edges,
                                         # edge_weight=-lap_norm,
                                         fill_value=1.,
                                         num_nodes=num_nodes)
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'ours':
            lxs = [h]
            edges, norm = add_self_loops(edge_index=lap_edges,
                                         # edge_weight=lap_norm,
                                         fill_value=-1.,
                                         num_nodes=num_nodes)
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'cheb':
            edges, norm = add_self_loops(edge_index=lap_edges,
                                         # edge_weight=lap_norm,
                                         fill_value=-1.,
                                         num_nodes=num_nodes)
            for k in range(self.K + 1):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm)
                else:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm) * 2
                    h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.poly == 'cheb2':
            #
            tlx = [h]
            edges, norm = add_self_loops(edge_index=lap_edges,
                                         # edge_weight=lap_norm,
                                         fill_value=-1.,
                                         num_nodes=num_nodes)
            for k in range(self.K):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm)

                else:
                    h = self.mp.propagate(edge_index=edges, x=h, norm=norm) * 2
                    h = h - tlx[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                tlx.append(h)
            #
            for j in range(self.K + 1):
                lxs.append(0)
                #
                xjs = []
                xj = math.cos((j + 0.5) * torch.pi / (self.K + 1))
                for i in range(self.K + 1):
                    if i == 0:
                        xjs.append(1)
                    elif i == 1:
                        xjs.append(xj)
                    else:
                        tmp = 2 * xj * xjs[-1] - xjs[-2]
                        xjs.append(tmp)
                    lxs[-1] = lxs[-1] + tlx[i] * xjs[-1]

        elif self.poly == 'bern':
            edges1, norm1 = lap_edges, lap_norm
            edges2, norm2 = add_self_loops(edge_index=lap_edges,
                                           # edge_weight=-lap_norm,
                                           fill_value=2.,
                                           num_nodes=num_nodes)
            tmps = [h]
            for k in range(self.K):
                h = self.mp.propagate(edge_index=edges1, x=h, norm=norm1)
                tmps.append(h)
            # all feature spaces
            for i, h in enumerate(tmps):
                tmp = h
                for j in range(self.K - i):
                    tmp = self.mp.propagate(edge_index=edges2, x=tmp, norm=norm2)
                tmp = tmp * comb(self.K, i) / 2 ** self.K
                lxs.append(tmp)

        #
        normed_lxs = []
        low_lxs = []
        for lx in lxs:
            if self.low_lx:
                U, S, V = svd_lowrank(lx)
                low_lx = torch.mm(U, torch.diag(S))
                low_lxs.append(low_lx)
                normed_lxs.append(F.normalize(low_lx, dim=1))
            else:
                normed_lxs.append(F.normalize(lx, dim=1))

        # final_lx = [F.normalize(lx, dim=1) for lx in normed_lxs] # norm1
        final_lx = [F.normalize(lx, dim=0) for lx in lxs]  # no norm1
        return final_lx

    def get_l_basis(self, edge_index, num_nodes, adj):
        if self.low_l:
            return adj
        # use adj
        l = torch.sparse_coo_tensor(indices=edge_index,
                                    values=torch.ones_like(edge_index[0]),
                                    size=(num_nodes, num_nodes),
                                    device=edge_index.device)
        # use lap
        lap_edges, lap_norm = get_laplacian(edge_index=edge_index,
                                            normalization='sym',
                                            num_nodes=num_nodes)
        l = torch.sparse_coo_tensor(indices=lap_edges,
                                    values=lap_norm,
                                    size=(num_nodes, num_nodes),
                                    device=edge_index.device).to_dense()
        if self.low_l:
            l = F.normalize(l, dim=1)
            U, S, V = svd_lowrank(l, q=self.nl)
            low_l = torch.mm(U, torch.diag(S))
            low_l = F.normalize(low_l, dim=0)
            return low_l
        else:
            l = F.normalize(l, dim=0)
            return l

class FEGNN(nn.Module):
    def __init__(self, ninput, noutput):
        super(FEGNN, self).__init__()
        self.K = 2
        self.poly = 'ours'
        # parser.add_argument("--poly", type=str, default='gpr', choices=['gpr', 'cheb', 'cheb2', 'bern', 'gcn', 'ours'])
        self.nx = ninput
        self.nlx = ninput
        self.nl = 0  # nl
        self.lin_x = nn.Linear(self.nx, 128, bias=True)
        self.lin_lx = nn.Linear(self.nlx, 128, bias=True)
        self.lin_l = nn.Linear(self.nl, 128, bias=True)
        self.lin2 = nn.Linear(128, noutput, bias=True)
        self.basis_generator = BasisGenerator(nx=self.nx, nlx=self.nlx, nl=self.nl, K=self.K, poly=self.poly,
                                              low_x=False, low_lx=False, low_l=True, norm1=False)
        self.thetas = nn.Parameter(torch.ones(self.K + 1), requires_grad=True)
        self.lin_lxs = torch.nn.ModuleList()
        for i in range(self.K + 1):
            self.lin_lxs.append(nn.Linear(self.nlx, 128, bias=True))
        self.share_lx = False

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        # x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:]  # []
        # l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict

        if self.nlx > 0:
            lx_dict = 0
            for k in range(self.K + 1):
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k]  # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k])  # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
            dict_mat = dict_mat + lx_dict

        # if self.nl > 0:
        #     l_dict = self.lin_l(l_basis)
        #     dict_mat = dict_mat + l_dict

        res = self.lin2(dict_mat)

        return F.log_softmax(res, dim=1)

    def get_dict(self, data):
        x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:]  # []
        l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0
        dict0 = []

        if self.nlx > 0:
            for k in range(self.K + 1):
                lx_dict = 0
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k]  # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k])  # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
                dict0.append(lx_basis[k])
            dict_mat = dict_mat + lx_dict

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict
            dict0.append(x_basis)

        if self.nl > 0:
            l_dict = self.lin_l(l_basis)
            dict_mat = dict_mat + l_dict
            dict0.append(l_basis)

        dict0 = torch.cat(dict0, dim=1)

        return dict0, dict_mat

class FEGNN(nn.Module):
    def __init__(self, ninput, noutput):
        super(FEGNN, self).__init__()
        self.K = 2
        self.poly = 'ours'
        # parser.add_argument("--poly", type=str, default='gpr', choices=['gpr', 'cheb', 'cheb2', 'bern', 'gcn', 'ours'])
        self.nx = ninput
        self.nlx = ninput
        self.nl = 0  # nl
        self.lin_x = nn.Linear(self.nx, 128, bias=True)
        self.lin_lx = nn.Linear(self.nlx, 128, bias=True)
        self.lin_l = nn.Linear(self.nl, 128, bias=True)
        self.lin2 = nn.Linear(128, noutput, bias=True)
        self.basis_generator = BasisGenerator(nx=self.nx, nlx=self.nlx, nl=self.nl, K=self.K, poly=self.poly,
                                              low_x=False, low_lx=False, low_l=True, norm1=False)
        self.thetas = nn.Parameter(torch.ones(self.K + 1), requires_grad=True)
        self.lin_lxs = torch.nn.ModuleList()
        for i in range(self.K + 1):
            self.lin_lxs.append(nn.Linear(self.nlx, 128, bias=True))
        self.share_lx = False

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        # x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:]  # []
        # l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict

        if self.nlx > 0:
            lx_dict = 0
            for k in range(self.K + 1):
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k]  # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k])  # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
            dict_mat = dict_mat + lx_dict

        # if self.nl > 0:
        #     l_dict = self.lin_l(l_basis)
        #     dict_mat = dict_mat + l_dict

        res = self.lin2(dict_mat)

        return F.log_softmax(res, dim=1)

    def get_dict(self, data):
        x, edge_index, cp_adj = data.x, data.edge_index, data.adj
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)[0:]  # []
        l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0], cp_adj)

        dict_mat = 0
        dict0 = []

        if self.nlx > 0:
            for k in range(self.K + 1):
                lx_dict = 0
                if self.share_lx:
                    lx_b = self.lin_lx(lx_basis[k]) * self.thetas[k]  # share W_lx across each layer/order
                else:
                    lx_b = self.lin_lxs[k](lx_basis[k])  # do not share the W_lx parameters
                lx_dict = lx_dict + lx_b
                dict0.append(lx_basis[k])
            dict_mat = dict_mat + lx_dict

        if self.nx > 0:
            x_dict = self.lin_x(x_basis)
            dict_mat = dict_mat + x_dict
            dict0.append(x_basis)

        if self.nl > 0:
            l_dict = self.lin_l(l_basis)
            dict_mat = dict_mat + l_dict
            dict0.append(l_basis)

        dict0 = torch.cat(dict0, dim=1)

        return dict0, dict_mat



class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,num_layers=3 ):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers

        # 初始化第一个GIN卷积层，它将原始特征维度转换为隐藏维度
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        ), train_eps=True))

        # 添加更多的GIN卷积层，每一层的输入和输出尺寸都是hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ), train_eps=True))

        # 最终的线性层，将最后一个GIN卷积层的输出转换为目标输出尺寸
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final_lin.reset_parameters()

    def forward(self, x, edge_index):
        # 对每个GIN卷积层进行迭代处理
        for conv in self.convs:
            x = conv(x, edge_index)

        # 应用最终的线性层，以得到正确的输出尺寸
        x = self.final_lin(x)

        return x

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout

        # 定义第一个GCN层，它将原始特征维度转换为隐藏维度
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # 添加更多的GCN层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 最终的GCN层，将特征维度转换为输出维度
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # 可选的批归一化层
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x) if len(self.bns) > i else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

from torch_geometric.nn import SAGEConv

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, dropout=0.5):
        super(SAGEEncoder, self).__init__()
        self.dropout = dropout

        # 定义第一个GraphSAGE层，它将原始特征维度转换为隐藏维度
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # 添加更多的GraphSAGE层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # 最终的GraphSAGE层，将特征维度转换为输出维度
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # 可选的批归一化层
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x) if len(self.bns) > i else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(FEGNN(in_channels, hidden_channels))
        self.convs.append(FEGNN(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()


    def forward(self, x, edge_index):
        # edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        # x = self.attention(x, x, x)
        return x

class FEGNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(FEGNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 添加第一层
        self.convs.append(FEGNN(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 添加中间层
        for _ in range(num_layers - 2):
            self.convs.append(FEGNN(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 添加最后一层
        self.convs.append(FEGNN(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        # 对于每一层，除了最后一层外，执行卷积，批量归一化，激活和dropout
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层不应用ReLU和Dropout
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EdgeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x

class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = torch.mul(x_1, x_2)
        return bi_layer

    def forward(self, h, edge):
        src_x = h[edge[0]]
        dst_x = h[edge[1]]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        #return torch.sigmoid(x)
        return x

class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(DegreeDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_channels, hidden_channels))
        self.mlps.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        return x


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss


class GMAE(nn.Module):
    def __init__(self, encoder, edge_decoder, degree_decoder, mask):
        super(GMAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = ce_loss
        self.negative_sampler = negative_sampling

    def train_epoch(self, data, optimizer, alpha, batch_size=8192, grad_norm=1.0):
        x, edge_index = data.x, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index)
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=masked_edges.view(2, -1).size(1)
        ).view_as(masked_edges)
        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            z = self.encoder(x, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)

            deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
            loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        # 预测正类和负类
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        # 合并预测结果和真实标签
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        pos_y = torch.ones(pos_pred.size(0))
        neg_y = torch.zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()

        # 计算性能指标
        auc = roc_auc_score(y, pred)
        aupr = average_precision_score(y, pred)
        binary_pred = (pred >= 0.5).astype(int)  # 二值化预测结果
        acc = accuracy_score(y, binary_pred)
        pre = precision_score(y, binary_pred)
        sen = recall_score(y, binary_pred)  # 召回率即灵敏度
        F1 = f1_score(y, binary_pred)
        mcc = matthews_corrcoef(y, binary_pred)

        # 返回所有计算出的指标和用于绘制ROC曲线的数据
        return auc, aupr, acc, pre, sen, F1, mcc, y, pred

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat", l2_normalize=False):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0))).cuda()
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        if l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)  # Cora, Citeseer, Pubmed

        return embedding


class GAE(nn.Module):
    def __init__(self, encoder, edge_decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

    def train_epoch(self, data, optimizer, alpha,batch_size=8192, grad_norm=1.0):
        self.train()
        x, edge_index = data.x.cuda(), data.edge_index.cuda()
        # 确保模型和所有输入数据都在同一设备上
        neg_edge_index = negative_sampling(edge_index, num_nodes=x.size(0),
                                           num_neg_samples=edge_index.size(1)).cuda()

        optimizer.zero_grad()
        z = self.encoder(x, edge_index)  # 编码获取节点嵌入
        edge_index_with_neg = torch.cat([edge_index, neg_edge_index], dim=1)  # 正样本和负样本边
        labels = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).cuda()

        # 预测边存在的概率
        preds = self.edge_decoder(z, edge_index_with_neg).squeeze()
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred
    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)

        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())
        # Recall和Precision计算
        recall = recall_score(y, temp.cpu())
        precision = precision_score(y, temp.cpu())
        # F1-Score计算
        F1 = f1_score(y, temp.cpu())

        return auc, ap, recall, pre, acc, F1

    def get_embedding(self, x, edge_index):
        self.eval()
        z = self.encoder(x, edge_index)
        return z


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.edge_decoder = Linear(out_channels * 2, 1)  # 用于边的存在概率预测

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def train_epoch(self, data, optimizer, device, alpha=0.007, batch_size=8192, grad_norm=1.0):
        self.train()
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        optimizer.zero_grad()

        z = self.forward(x, edge_index)
        pos_edge_index, neg_edge_index = data.pos_edge_label_index.to(device), data.neg_edge_label_index.to(device)
        pos_out = self.decode(z, pos_edge_index)
        neg_out = self.decode(z, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_out, neg_out]).squeeze(), torch.cat(
            [torch.ones(pos_out.size(0), device=device), torch.zeros(neg_out.size(0), device=device)]))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
        optimizer.step()

        return loss.item()

    def decode(self, z, edge_index):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        edge_probs = self.edge_decoder(edge_features)
        return edge_probs

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index, device):
        z = z.to(device)
        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)
        pos_pred = self.decode(z, pos_edge_index).sigmoid()
        neg_pred = self.decode(z, neg_edge_index).sigmoid()

        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()  # 注意转换回numpy前需移至CPU
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu().numpy()

        auc = roc_auc_score(y_true, pred)
        ap = average_precision_score(y_true, pred)

        # 直接使用numpy计算其他指标
        temp = pred >= 0.5
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y_true, temp)  # 确保calculate_metrics可以处理numpy数组
        return auc, ap, recall_score(y_true, temp), precision_score(y_true, temp), acc, F1

    def get_embedding(self, x, edge_index, device):
        self.eval()
        x = x.to(device)
        edge_index = edge_index.to(device)
        z = self.forward(x, edge_index)
        return z
