import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Path_MGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(Path_MGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.PGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.EGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )

    def forward(self, x, fadj, padj1, padj2, sadj):
        emb1 = self.FGCN(x, fadj)  # Feature_GCN
        emb2 = self.PGCN(x, padj1)  # pathway_activity_GCN
        emb3 = self.EGCN(x, padj2)  # pathway_expression_GCN
        emb4 = self.SGCN(x, sadj)  # Spatial_GCN

        com1 = self.CGCN(x, fadj)  # Co_GCN
        com2 = self.CGCN(x, padj1)  # Co_GCN
        com3 = self.CGCN(x, padj2)  # Co_GCN
        com4 = self.CGCN(x, sadj)  # Co_GCN

        emb = torch.stack([emb2, (com2 + com4) / 2, emb4], dim=1)
        emb, att = self.att(emb)
        emb = self.MLP(emb)

        [pi, disp, mean] = self.ZINB(emb)
        return com2, com4, emb, pi, disp, mean

    # def forward(self, x, padj, sadj):
    #     emb2 = self.PGCN(x, padj)  # pathway_activity_GCN
    #     emb4 = self.SGCN(x, sadj)  # Spatial_GCN
    #
    #     com2 = self.CGCN(x, padj)  # Co_GCN
    #     com4 = self.CGCN(x, sadj)  # Co_GCN
    #
    #     # emb = torch.stack([emb2, (com2 + com3 + com4) / 3, emb4], dim=1)
    #     emb = torch.stack([emb2, (com2 + com4) / 2, emb4], dim=1)
    #     emb, att = self.att(emb)
    #     emb = self.MLP(emb)
    #
    #     [pi, disp, mean] = self.ZINB(emb)
    #     return com2, com4, emb, pi, disp, mean

