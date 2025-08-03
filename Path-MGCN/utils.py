import os
import random
import tempfile
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import community as community_louvain
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import scanpy as sc
import h5py
from config import Config
from models import Path_MGCN

def get_pathway_activity_with_r(adata, gmt_file, path, r_script="GSVA_scores.R", threads=80):

    if os.path.exists(path + 'pathway_activity.csv'):
        pathway_activity = pd.read_csv(path + 'pathway_activity.csv', index_col=0)
        print('Pathway activity file already exists')
        return pathway_activity

    expr_df = pd.DataFrame(
        adata.X.toarray().T if sp.issparse(adata.X) else adata.X.T,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str)
    )
    tf_expr = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    expr_df.to_csv(tf_expr.name, sep="\t")

    tf_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    tf_out.close()

    cmd = [
        "Rscript", r_script,
        "--expr", tf_expr.name,
        "--gmt",  gmt_file,
        "--out",  tf_out.name,
        "--threads", str(threads)
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tf_expr.name)

    pathway_activity = pd.read_csv(tf_out.name, sep="\t", index_col=0)
    os.remove(tf_out.name)
    pathway_activity = pathway_activity.loc[adata.obs_names].astype("float32")
    pathway_activity.to_csv(path + 'pathway_activity.csv')

    return pathway_activity


def prepare_data_in_memory(dataset, highly_genes, k, radius, gmt_file, path, threads):
    print(f"--- Starting data processing for dataset: {dataset} ---")

    adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    labels_df = pd.read_table(os.path.join(path, "metadata.tsv"), sep='\t')
    labels_df.index = adata.obs.index
    adata.obs['ground_truth'] = labels_df["layer_guess_reordered"]
    adata = adata[~adata.obs['ground_truth'].isnull()].copy()

    print("Normalizing gene expression and selecting highly variable genes...")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    pathway_activity = get_pathway_activity_with_r(adata, gmt_file, path, threads=threads)
    adata.obsm['pathway_activity'] = pathway_activity.loc[adata.obs_names].values

    print("Constructing spatial and pathway graphs...")
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph(adata, radius=radius)
    padj = features_construct_graph(adata.obsm['pathway_activity'], k=k)

    features = torch.FloatTensor(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    labels = adata.obs['ground_truth'].values

    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    npadj = normalize_sparse_matrix(padj + sp.eye(padj.shape[0]))
    npadj = sparse_mx_to_torch_sparse_tensor(npadj)

    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    graph_nei_tensor = torch.LongTensor(graph_nei.numpy())
    graph_neg_tensor = torch.LongTensor(graph_neg.numpy())

    print("--- Data preparation complete ---")
    return adata, features, labels, nfadj, npadj, nsadj, graph_nei_tensor, graph_neg_tensor


def run_training(model, optimizer, features, fadj, padj, sadj, graph_nei, graph_neg, config):
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    model.train()
    optimizer.zero_grad()

    com_pathway, com_spatial, emb, pi, disp, mean = model(features, fadj, padj, padj, sadj)

    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com_pathway, com_spatial)

    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    total_loss.backward()
    optimizer.step()
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    return emb, total_loss.item()

def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def spatial_construct_graph(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg

def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result

def normalize_sparse_matrix(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx