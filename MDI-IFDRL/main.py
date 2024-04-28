import torch
import numpy as np
import argparse
from mask import *
from utils import get_data, set_seed, prepare_cross_validation_data
from model import GNNEncoder, FEGNNEncoder,EdgeDecoder, DegreeDecoder, GMAE, GAE, GraphSAGE, GINEncoder, GATEncoder, GCNEncoder, LPDecoder,SAGEEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, average_precision_score

# 准备绘制ROC曲线的设置
plt.figure(figsize=(10, 8))
# main parameter
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help="Random seed for model and dataset.")
parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_layers', type=int, default=2)
args = parser.parse_args()
set_seed(args.seed)

# {'AUC': '0.985314', 'AP': '0.980734', 'ACC': '0.949944', 'SEN': '0.975528', 'PRE': '0.928042', 'SPE': '0.924360', 'F1': '0.951193', 'MCC': '0.901069'}
encoder = GCNEncoder(in_channels=1177, hidden_channels=64, out_channels=128)
lpencoder = FEGNNEncoder(in_channels=1177, hidden_channels=64, out_channels=128,num_layers=args.num_layers,
                     dropout=args.dropout)
edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
lp_decoder = LPDecoder(in_channels=128, hidden_channels=64, out_channels=1, encoder_layer=1, num_layers=args.num_layers,
                     dropout=args.dropout)
degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
mask = MaskPath(p=args.p)
best_metrics = {}
best_auc = 0.0
model = GMAE(encoder, lp_decoder, degree_decoder, mask).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# splits = get_data()
splits_data = prepare_cross_validation_data()  # 准备十折交叉验证的数据
fold_metrics = {'AUC': [], 'AUPR': [], "Acc": [], "Pre": [], "SEN": [], "F1": [], "MCC": []}
# 存储所有折的TPR和FPR，用于计算平均ROC曲线
all_tpr = []
all_fpr = np.linspace(0, 1, 100)  # 定义一个用于插值的公共FPR

# 存储每一折的AUC和AUPRC
fold_aucs = []
fold_auprcs = []
# 用于存储所有折的真阳性率（TPR）值
mean_tpr = np.zeros_like(all_fpr)
num_folds = len(splits_data)  # 获取折数
for fold_index, splits in enumerate(splits_data):
    epoch_metrics_sum = {'AUC': 0, 'AUPR': 0, "Acc": 0, "Pre": 0, "SEN": 0, "F1": 0, "MCC": 0}
    for epoch in range(1000):
        model.train()
        train_data = splits['train'].cuda()
        x, edge_index = train_data.x, train_data.edge_index
        loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
        model.eval()
        test_data = splits['test'].cuda()
        z = model.encoder(test_data.x, test_data.edge_index)

        # test_auc, test_aupr, acc, pre, sen, F1, mcc = model.test(z, test_data.pos_edge_label_index,
        #                                                          test_data.neg_edge_label_index)
        test_auc, test_aupr, acc, pre, sen, F1, mcc, y_true, y_scores = model.test(z, test_data.pos_edge_label_index,
                                                                                   test_data.neg_edge_label_index)

        # 根据新的返回值更新results字典，包括SEN和MCC
        results = {
            'AUC': "{:.6f}".format(test_auc),
            'AUPR': "{:.6f}".format(test_aupr),
            "Acc": "{:.6f}".format(acc),
            "Pre": "{:.6f}".format(pre),
            "SEN": "{:.6f}".format(sen),  # 使用SEN代替Recall
            "F1": "{:.6f}".format(F1),
            "MCC": "{:.6f}".format(mcc),  # 新增MCC
        }

        print(results)
        fold_metrics['AUC'].append(test_auc)
        fold_metrics['AUPR'].append(test_aupr)
        fold_metrics['Acc'].append(pre)
        fold_metrics['Pre'].append(test_auc)
        fold_metrics['SEN'].append(sen)
        fold_metrics['F1'].append(F1)
        fold_metrics['MCC'].append(mcc)

        for metric in epoch_metrics_sum:
            # 将结果字符串转换为浮点数，然后累加
            epoch_metrics_sum[metric] += float(results[metric])

            # 计算并存储每一折的平均指标
    fold_avg_metrics = {metric: epoch_metrics_sum[metric] / 1000 for metric in epoch_metrics_sum}
    print(f"Fold {fold_index + 1} Average Metrics:")
    for metric, avg_value in fold_avg_metrics.items():
        print(f"{metric}: {avg_value:.6f}")
        fold_metrics[metric].append(avg_value)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Fold {fold_index + 1} (AUC = {roc_auc:.4f})')
    # 插值计算并更新平均 TPR
    mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

# 完成所有折后计算平均 TPR
mean_tpr /= num_folds
mean_tpr[-1] = 1.0  # 确保最后一个 TPR 是 1
# 计算所有折的平均指标
avg_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
# 计算平均 AUC
mean_auc = auc(all_fpr, mean_tpr)

# 绘制平均 ROC 曲线
plt.plot(all_fpr, mean_tpr, color='red',
         label=f'Mean ROC ', lw=2, alpha=.8)

# 绘图设置

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for each fold and Mean Curve')
plt.legend(loc="lower right")

# 显示图形
plt.show()

# 打印所有折的平均指标
print("\nAverage Metrics Across All Folds:")
for metric, avg_value in avg_metrics.items():
    print(f"{metric}: {avg_value:.6f}")
    #     # 比较和更新最高指标
    #     if test_auc > best_auc:
    #         best_auc = test_auc
    #         best_metrics = results
    #
    # # 循环结束后打印最高 AUC 对应的指标
    # print("Best Metrics with highest AUC:")
    # for key, value in best_metrics.items():
    #     try:
    #         value = float(value)  # 尝试将值转换为浮点数
    #         print(f"{key}: {value:.6f}")
    #     except ValueError:
    #         print(f"{key}: {value}")  # 如果转换失败，直接打印原始字符串