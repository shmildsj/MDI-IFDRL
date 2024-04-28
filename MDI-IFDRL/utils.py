import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score, f1_score
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.model_selection import StratifiedKFold
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred_proba):
    y_pred = np.round(y_pred_proba)  # 根据概率值得到二元预测

    auc = roc_auc_score(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return auc, aupr, recall, precision, accuracy, f1


def get_data():
    interaction = pd.read_csv("dataset/MDA/MD_A.csv", index_col=0)
    d_feature = np.loadtxt("dataset/MDA/DD.txt")
    m_feature = np.loadtxt("dataset/MDA/MM.txt")

    m_emb = torch.FloatTensor(d_feature)
    print(m_emb.size())
    s_emb = torch.FloatTensor(m_feature)
    print(s_emb.size())
    m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - m_emb.size(1))], dim=1)
    s_emb = torch.cat([s_emb, torch.zeros(s_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - s_emb.size(1))], dim=1)

    feature = torch.cat([m_emb, s_emb]).cuda()

    l, p = interaction.values.nonzero()
    adj = torch.LongTensor([p, l + len(d_feature)]).cuda()
    data = Data(x=feature, edge_index=adj).cuda()
    print(data)
    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)
    splits = dict(train=train_data, test=test_data)
    return splits


def prepare_cross_validation_data(k_folds=10):
    interaction = pd.read_csv("dataset/MDA/MD_A.csv", index_col=0)
    d_feature = np.loadtxt("dataset/MDA/DD.txt")
    m_feature = np.loadtxt("dataset/MDA/MM.txt")

    m_emb = torch.FloatTensor(d_feature)
    print(m_emb.size())
    s_emb = torch.FloatTensor(m_feature)
    print(s_emb.size())
    m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - m_emb.size(1))], dim=1)
    s_emb = torch.cat([s_emb, torch.zeros(s_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - s_emb.size(1))], dim=1)

    feature = torch.cat([m_emb, s_emb]).cuda()

    l, p = interaction.values.nonzero()
    adj = torch.LongTensor([p, l + len(d_feature)]).cuda()
    data = Data(x=feature, edge_index=adj).cuda()

    edges = data.edge_index.t().cpu().numpy()  # 转换为NumPy数组
    edge_labels = np.ones((edges.shape[0],))  # 正样本标签，这里只是作为形式参数

    # 生成负样本
    num_nodes = data.x.size(0)
    num_neg_samples = edges.shape[0]  # 与正样本数量相同
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j and [i, j] not in edges.tolist() and [i, j] not in neg_edges:
            neg_edges.append([i, j])
    neg_edges = np.array(neg_edges)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_data = []
    for train_idx, test_idx in skf.split(edges, edge_labels):
        # 分割训练集和测试集的正样本边
        train_edges = edges[train_idx]
        test_edges = edges[test_idx]

        # 负样本（简单分割）
        train_neg_edges = neg_edges[train_idx]
        test_neg_edges = neg_edges[test_idx]

        # 直接创建每个fold的训练和测试Data对象，不包括负样本
        train_data = Data(x=data.x, edge_index=torch.tensor(train_edges).t().contiguous().to(torch.long),
                          pos_edge_label_index=torch.tensor(train_edges).t().contiguous().to(torch.long),
                          neg_edge_label_index=torch.tensor(train_neg_edges).t().contiguous().to(torch.long))
        test_data = Data(x=data.x, edge_index=torch.tensor(test_edges).t().contiguous().to(torch.long),
                         pos_edge_label_index=torch.tensor(test_edges).t().contiguous().to(torch.long),
                         neg_edge_label_index=torch.tensor(test_neg_edges).t().contiguous().to(torch.long)
                         )

        fold_data.append(dict(train=train_data, test=test_data))
    return fold_data


if __name__ == '__main__':
    data = get_data(2, 2048)
