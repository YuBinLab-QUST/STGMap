import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.neighbors
from numpy import random
from sklearn.metrics import roc_curve
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.metrics  import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# ================= Data importing =================   
# ==================================================
def filter_gene(df, filter_num):
    rows_nonzero = []
    cols_nonzero = []
    for i in range(df.values.shape[0]):
        rows_nonzero.append(np.where(df.values[i,:]!=0)[0].shape[0])

    for i in range(df.values.shape[1]):
        cols_nonzero.append(np.where(df.values[:,i]!=0)[0].shape[0])

    # result = Counter(cols_nonzero)

    not_filtered_gene = []
    for i in df.columns.tolist():
        if np.where(df[i].values!=0)[0].shape[0] >= filter_num:
            not_filtered_gene.append(i)

    return df[not_filtered_gene]


def log_transform(df, add_number):
    df_add = df + add_number
    df_log = np.log(df_add)
    df_log_zscore = StandardScaler().fit_transform(df_log).T
    df_log_zscore = StandardScaler().fit_transform(df_log_zscore).T
    return pd.DataFrame(df_log_zscore, columns=df_add.columns)


def read_dataset(input_exp, input_adj, filter_num=None, add_number=None):
    exp = pd.read_csv(open(input_exp))
    adj = pd.read_csv(open(input_adj))
#    if not (adj.values==adj.values.T).all():
#        raise ImportError('The input adjacency matrix is not a symmetric matrix, please check the input.')
    if not np.diag(adj.values).sum()==0:
        raise ImportError('The diagonal elements of input adjacency matrix are not all 0, please check the input.')
    if filter_num is not None:
        exp = filter_gene(exp, filter_num)
    if add_number is not None:
        exp = log_transform(exp, add_number)

    return exp, adj


def read_coordinate(input_coord):
    coord = pd.read_csv(open(input_coord))  
    return coord


def read_cell_label(input_label):
    label = pd.read_csv(open(input_label))  
    return label


def write_csv_matrix(matrix, filename, ifindex=False, ifheader=True, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
        ifindex, ifheader = ifheader, ifindex

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename+'.csv', index=ifindex, header=ifheader)


# =============== Data processing ===============
# ===============================================
def Adj_generation(exp_df, coord_df):
    k_cutoff=6
    coord_df.index = exp_df.index
    coord_df.columns = ['imagerow', 'imagecol']
    
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coord_df)
    distances, indices = nbrs.kneighbors(coord_df)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coord_df.shape[0]), np.array(coord_df.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    cells = np.array(exp_df.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    Adj = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(cells.shape[0], cells.shape[0]))
    Adj = Adj.todense()
    Adj = np.asarray(Adj)
    Adj = Adj.astype(int)
    return Adj


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def pack_dataset(adj, exp, cell_label):     
    A = torch.tensor(adj)
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = normalize_graph(A_I)
    A_I_nomal = A_I_nomal.to_sparse()
              
    x = torch.FloatTensor(exp)
  
    adj1 = sp.coo_matrix(adj)
    indices = np.vstack((adj1.row, adj1.col))
    edge_index = torch.LongTensor(indices)
    data = Data(x=x, edge_index=edge_index)
        
    eps = 2.2204e-16
    norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    data.x = data.x.div(norm.expand_as(data.x))
    adj_1 = sp.csr_matrix(adj)

    lable = cell_label[:,1]
    lable = np.array(lable,dtype=int)
    lable = torch.tensor(lable)
    nb_feature = data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1
    nb_nodes = data.num_nodes
    return data, [A_I_nomal,adj_1], [data.x], [lable, nb_feature, nb_classes, nb_nodes]


def ismember(tmp1, tmp2, tol=5):
    rows_close = np.all(np.round(tmp1 - tmp2[:, None], tol) == 0, axis=-1)
    if True in np.any(rows_close, axis=-1).tolist():
        return True
    elif True not in np.any(rows_close, axis=-1).tolist():
        return False


def sparse2tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def get_aftermiss_adj(adj, miss_ratio):
    miss_ratio=0.9
    adj = sp.csr_matrix(adj)
    adj_values = adj
    edges_single = sparse2tuple(sp.triu(adj_values))[0]
    num_miss = int(np.floor(edges_single.shape[0] * miss_ratio))
    all_edges_idx = list(range(edges_single.shape[0]))
    np.random.shuffle(all_edges_idx)
    miss_edges_idx = all_edges_idx[:num_miss]
    after_edges = np.delete(edges_single, miss_edges_idx, axis=0)
    miss_edges = edges_single[miss_edges_idx,:]
    
    data = np.ones(after_edges.shape[0])
    adj_after = sp.csr_matrix((data, (after_edges[:, 0], after_edges[:, 1])), shape=adj_values.shape)
    adj_after = adj_after + adj_after.T
    
    data1 = np.ones(miss_edges.shape[0])
    adj_miss = sp.csr_matrix((data1, (miss_edges[:, 0], miss_edges[:, 1])), shape=adj_values.shape)
    adj_miss = adj_miss + adj_miss.T

    adj_after = adj_after.todense()
    adj_after = np.asarray(adj_after)
    adj_after = adj_after.astype(int)
    adj_miss = adj_miss.todense()
    adj_miss = np.asarray(adj_miss)
    adj_miss = adj_miss.astype(int)
    return adj_after, adj_miss


def get_afteradd_adj(adj, add_ratio):
    add_ratio = 3
    adj_values = adj
    edges_single = sparse2tuple(sp.triu(adj_values))[0]
    num_add = int(np.floor(edges_single.shape[0] * add_ratio))
    add_edges = random.randint(0, adj.shape[0], size=(num_add,2))
    merge_edges = np.append(edges_single,add_edges,axis=0)
    merge_edges = np.unique(merge_edges, axis=0)

    data = np.ones(merge_edges.shape[0])
    adj_after = sp.csr_matrix((data, (merge_edges[:, 0], merge_edges[:, 1])), shape=adj_values.shape)
    adj_after = adj_after + adj_after.T

    adj_after = adj_after.todense()
    adj_after = np.asarray(adj_after)
    adj_after = adj_after.astype(int)
    adj_after[adj_after > 0.0] = 1.0
    return adj_after  


# =============== Training and testing set splitting  ===============
# ===================================================================
def sampling_test_edges_neg(n, test_edges, edges_double):
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, n)
        idx_j = np.random.randint(0, n)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_double):
            continue
        if ismember([idx_j, idx_i], edges_double):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    return test_edges_false


def train_test_split(adj_values, test_ratio=0.1):
    adj_values = sp.csr_matrix(adj_values)
    edges_single = sparse2tuple(sp.triu(adj_values))[0]
    edges_double = sparse2tuple(adj_values)[0]

    if test_ratio > 1:
        test_ratio = test_ratio/edges_single.shape[0]

    num_test = int(np.floor(edges_single.shape[0] * test_ratio))
    all_edges_idx = list(range(edges_single.shape[0]))
    np.random.shuffle(all_edges_idx)
    test_edges_idx = all_edges_idx[:num_test]
    test_edges = edges_single[test_edges_idx]
    if (adj_values.shape[0]**2-adj_values.sum()-adj_values.shape[0])/2 < 2*len(test_edges):
        raise ImportError('The network is too dense, please reduce the proportion of test set or delete some edges in the network.')
    else:
        test_edges_false = sampling_test_edges_neg(adj_values.shape[0], test_edges, edges_double)
    train_edges = np.delete(edges_single, test_edges_idx, axis=0)

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj_values.shape)
    adj_train = adj_train + adj_train.T
    data = np.ones(test_edges.shape[0])
    adj_test = sp.csr_matrix((data, (test_edges[:, 0], test_edges[:, 1])), shape=adj_values.shape)
    adj_test = adj_test + adj_test.T

    return adj_train, adj_test, train_edges, test_edges, test_edges_false


# =============== Model running ===============
# ===============================================
class linkpred_metrics():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            if x >= 0:
                return 1.0/(1+np.exp(-x))
            else:
                return np.exp(x)/(1+np.exp(x))

        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        acc_score = accuracy_score(labels_all, np.round(preds_all))

        fpr, tpr, threshold1s = roc_curve(labels_all, preds_all)

        return roc_score, ap_score, acc_score, emb, fpr

