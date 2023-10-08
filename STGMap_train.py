import warnings
import argparse
warnings.filterwarnings("ignore")
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from utils import *
from models import *



def parse_args():
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--seed", default=0)
    # Dataset
    parser.add_argument('--exp', '-e', type=str, help='TODO: Input gene expression data path')
    parser.add_argument('--adj', '-a', type=str, help='Input adjacency matrix data path')
    parser.add_argument('--coordinate', '-c', type=str, help='Input cell coordinate data path')
    parser.add_argument('--reference', '-r', type=str, help='Input cell type label path')
    # Pretrain
    parser.add_argument("--pretrain", default=True, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)
    parser.add_argument('--w_loss1', type=float, default=10, help='')
    parser.add_argument('--w_loss2', type=float, default=10, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    return parser.parse_args()


# ===================================================#
args = parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
running_device = "cpu" 
# ===================================================#
args.exp = "d:\\Users\\DELL\\Desktop\\DeepLinc\\DeepLinc-main\\dataset\\seqFISH\\counts.csv"
args.adj = "d:\\Users\\DELL\\Desktop\\DeepLinc\\DeepLinc-main\\dataset\\seqFISH\\adj.csv"
args.fil_gene = None
args.log_add_number = None
args.coordinate = "d:\\Users\\DELL\\Desktop\\DeepLinc\\DeepLinc-main\\dataset\\seqFISH\\coord.csv"
args.reference = "d:\\Users\\DELL\\Desktop\\DeepLinc\\DeepLinc-main\\dataset\\seqFISH\\cell_type_4.csv"
# ===================================================#
exp_df, adj_df = read_dataset(args.exp, args.adj, args.fil_gene, args.log_add_number)
exp, adj = exp_df.values, adj_df.values
coord_df = read_coordinate(args.coordinate)
coord = coord_df.values
cell_label_df = read_cell_label(args.reference)
cell_label = cell_label_df.values
# ===================================================#
adj_train, adj_test, train_edges, test_edges, test_edges_false = train_test_split(adj, test_ratio=0.1)
adj = adj_train.todense()
adj = np.asarray(adj)
adj = adj.astype(int)
# ===================================================#
data, adj_list, x_list, nb_list = pack_dataset(adj, exp, cell_label)
lable = nb_list[0]
nb_feature = nb_list[1]
nb_classes = nb_list[2]
nb_nodes = nb_list[3]
feature_X = x_list[0].to(running_device)
A_I_nomal = adj_list[0].to(running_device)
# ===================================================#
cfg = [512,128]
dropout = 0.1
model = STGMap(nb_feature, cfg, dropout)
weight_decay = 0.0001
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
model.to(running_device)
lable = lable.to(running_device)
# ===================================================#
my_margin = args.margin1
my_margin_2 = my_margin + args.margin2
margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
num_neg = 4
lbl_z = torch.tensor([0.]).to(running_device)

A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
edge_index = A_I_nomal._indices()[1]

deg_list_2 = []
deg_list_2.append(0)
for i in range(nb_nodes):
    deg_list_2.append(deg_list_2[-1] + A_degree[i])
idx_p_list = []
for j in range(1, 101):
    random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
    idx_p = edge_index[random_list]
    idx_p_list.append(idx_p)

train_loss = []
test_ap = []
test_roc = []
test_acc = []
test_fpr = []
max_test_ap_score = 0
for epoch in range(500):
    optimiser.zero_grad()   
    idx_list = []     
    for i in range(num_neg):  
        idx_0 = np.random.permutation(nb_nodes)
        idx_list.append(idx_0)
    h_a, h_p = model(feature_X, A_I_nomal)
    embs = h_p
    embs = embs / embs.norm(dim=1)[:, None]
    embs = embs.detach().numpy()
    embs_latent = 10**embs
    h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
         idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                  idx_p_list[(epoch + 8) % 100]]) / 5
    s_p = F.pairwise_distance(h_a, h_p)
    s_p_1 = F.pairwise_distance(h_a, h_p_1)
    s_n_list = []
    for h_n in idx_list:
        s_n = F.pairwise_distance(h_a, h_a[h_n])
        s_n_list.append(s_n)     
    margin_label = -1 * torch.ones_like(s_p)

    loss_mar = 0
    loss_mar_1 = 0
    mask_margin_N = 0
    for s_n in s_n_list:
        loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
        loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
        mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
    mask_margin_N = mask_margin_N / num_neg

    loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3
    loss.backward()
    optimiser.step()
     
    train_loss.append(loss)
    lm_train = linkpred_metrics(test_edges, test_edges_false)
    roc_score, ap_score, acc_score, _, fpr_score = lm_train.get_roc_score(embs)
    test_ap.append(ap_score)
    test_roc.append(roc_score)
    test_acc.append(acc_score)
    fpr_score = np.nanmean(fpr_score,axis=0)
    test_fpr.append(fpr_score)

    print("Epoch:", '%04d' % (epoch + 1), "loss_1=", "{:.3f}".format(loss_mar.item()), "loss_2=", "{:.3f}".format(loss_mar_1.item()), "loss_3=", "{:.3f}".format(mask_margin_N.item()), "loss=", "{:.3f}".format(loss.item()), embs)

