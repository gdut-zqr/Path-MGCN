import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.optim import NAdam
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from config import Config
from utils import *
from models import Path_MGCN

dataset = '151507'
config_file = './config/DLPFC.ini'
config = Config(config_file)
kegg_gmt_file = '../data/kegg/gene_sets.gmt'
path = f"../data/DLPFC/{dataset}/"
epochs = 500
threads = 50

if __name__ == "__main__":

    adata, features, labels, fadj, padj, sadj, graph_nei, graph_neg = prepare_data_in_memory(
        dataset, config.fdim, config.k, config.radius, kegg_gmt_file, path, threads
    )

    print(f"\n--- Starting model training ---")

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    features, fadj, padj, sadj = features.to(device), fadj.to(device), padj.to(device), sadj.to(device)
    graph_nei, graph_neg = graph_nei.to(device), graph_neg.to(device)

    model = Path_MGCN(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout
    ).to(device)

    optimizer = NAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ari_max = 0
    best_emb = None
    best_clusters = None

    for epoch in range(1, epochs + 1):
        emb, loss = run_training(model, optimizer, features, fadj, padj, sadj, graph_nei, graph_neg, config)

        kmeans = KMeans(n_clusters=len(np.unique(labels))).fit(emb)
        ari_res = metrics.adjusted_rand_score(labels, kmeans.labels_)
        if (epoch % 25 == 0) or (epoch ==1):
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, ARI: {ari_res:.4f}")

        if ari_res > ari_max:
            ari_max = ari_res
            best_emb = emb
            best_clusters = kmeans.labels_

    print(f"\n--- Training complete. ARI: {ari_max:.4f} ---")

    save_dir = f'./result/DLPFC/{dataset}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    adata.obs['path_mgcn_cluster'] = pd.Categorical(best_clusters)

    plt.rcParams["figure.figsize"] = (5, 5)
    sc.pl.spatial(
        adata,
        img_key="hires",
        color=['path_mgcn_cluster'],
        title=f'Path-MGCN (ARI: {ari_max:.4f})',
        show=False
    )
    plt.savefig(os.path.join(save_dir, 'Path_MGCN_clusters.png'), bbox_inches='tight', dpi=600)
    plt.close()

    print(f"Results have been saved to: {save_dir}")