import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from utils import *
from config import args
from torch_geometric.nn import GATConv

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.pool1 = nn.AvgPool1d(4)  # d/4 dim
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=1,
            padding=1,
            dilation=2
        )
        self.pool2 = nn.AvgPool1d(4)  # d/16 dim
        self.conv = nn.Sequential(
            self.conv1, self.pool1, nn.Tanh(),
            self.conv2, self.pool2, nn.Tanh()
        )
        self.lin = nn.Linear(args.d // 16, args.c)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.lin(x)
        x = F.sigmoid(x)
        return x


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.conv1 = nn.LSTM(
            input_size=args.d,
            hidden_size=args.hidden,
        )
        self.conv2 = nn.LSTM(
            input_size=args.hidden,
            hidden_size=args.hidden,
        )
        self.lin = nn.Linear(args.hidden, args.c)

    def forward(self, x):
        x = F.tanh(self.conv1(x)[0])
        x = F.tanh(self.conv2(x)[0])
        x = x.squeeze()
        x = F.sigmoid(self.lin(x))
        return x


class GraphConv(nn.Module):
    # my implementation of GCN
    def __init__(self, in_dim, out_dim, drop=0.5, bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.zeros(1, out_dim))
            nn.init.xavier_uniform_(self.b)

    def forward(self, adj, x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


class GCNNet(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, hid_dim=args.hidden, bias=True):
        super(GCNNet, self).__init__()
        self.res1 = GraphConv(in_dim, hid_dim, bias=bias, activation=F.relu)
        self.res2 = GraphConv(hid_dim, hid_dim, bias=bias, activation=F.relu)
        self.res3 = GraphConv(hid_dim, out_dim, bias=bias, activation=F.sigmoid)

    def forward(self, g, z):  # g: adj, z: feature
        h = self.res1(g, z)
        h = self.res2(g, h)
        output = self.res3(g, h)
        return output



class GATNet(nn.Module):
    def __init__(self, in_dim=args.d, out_dim=args.c, heads=args.heads, hid_dim=args.hidden, bias=True):
        super(GATNet, self).__init__()
        self.res1 = GATConv(in_dim, hid_dim, heads, bias=bias, concat=False)
        self.res2 = GATConv(hid_dim, hid_dim, heads, bias=bias, concat=False)
        self.res3 = GATConv(hid_dim, out_dim, heads, bias=bias, concat=False)

    def forward(self, g, z):  # g: adj, z: feature
        edge_index = adjacency_to_edge_index(g)
        h = self.res1(z, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.res2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        output = self.res3(h, edge_index)
        return output