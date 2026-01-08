import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import leaky_relu
import numpy as np
import math

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Encoder(nn.Module):
    def __init__(self, D, d, h_dims=[64], last_act=True):
        super().__init__()
        self.net = self.make_mlp([D] + h_dims + [d], last_act=last_act)

    def make_mlp(self, mlp_dims, activation="leaky_relu", last_act=False):
        layers = []
        mlp_dims = mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        net = nn.Sequential(*layers)
        return net

    def forward(self, data):
        out = self.net(data)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        # Positional Encoding行列の生成
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        # バッチ次元と歩行者次元に合わせて拡張
        self.pe = self.pe.unsqueeze(0).unsqueeze(2)  # (1, max_len, 1, d_model)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :, :].to(x.device)
        return x
  
class GCLayer(nn.Module):
    def __init__(self, d, D, device="cpu"):
        super(GCLayer, self).__init__()
        self.d = d
        self.D = D
        self.device = device
        self.W = nn.Parameter(torch.randn(d, D))

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, X, A):
        X_gc = torch.matmul(A, X)

        gc = torch.matmul(X_gc, self.W)
        out = leaky_relu(gc)
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_heads=8,
        dropout_rate=0.0,
        alpha=0.2,
        concat=True,
        share_weights=False,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        if concat:
            assert out_features % n_heads == 0
            self.h_dim = out_features // n_heads
        else:
            self.h_dim = out_features

        self.linear_l = nn.Linear(in_features, self.h_dim * n_heads, bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.h_dim * n_heads, bias=False)

        self.attn = nn.Linear(self.h_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h):
        #program : example detail
        n_batches = h.shape[0] #batch size 100
        n_nodes = h.shape[1] #human num 5

        g_l = self.linear_l(h).view(n_batches, n_nodes, self.n_heads, self.h_dim) # [100,5,32]->Linear(32,32)->[100,5,8,4]
        g_r = self.linear_r(h).view(n_batches, n_nodes, self.n_heads, self.h_dim) # [100,5,32]->Linear(32,32)->[100,5,8,4]

        #to calculate [g_i||g_j] for all pairs
        #[g_1,g_2,...g_N]*n_nodes
        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1) # [100,5,8,4]*5->[100,25,8,4] repeat=1->same tensor
        #[g_1,g_1,...,g_2,g_2,...g_N,g_N...]
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1) # [100,5,8,4]*5->[100,25,8,4]

        g_sum = g_l_repeat + g_r_repeat_interleave
        #Reshape so that g_sum[i, j] is gl​i​+gr​j​
        g_sum = g_sum.view(n_batches, n_nodes, n_nodes, self.n_heads, self.h_dim)#[100,25,8,4]->[100,5,5,8,4]

        #apply Linear(4,1)for each g_sum
        e = self.attn(self.activation(g_sum))#[100,5,5,8,1]

        e = e.squeeze(-1)#[100,5,5,8]

        a = self.softmax(e)

        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g_r)#[100,5,8,4]
        
        if self.concat:
            return attn_res.reshape(n_batches, n_nodes, self.n_heads * self.h_dim)
        else:
            return attn_res.mean(dim=1)
        

class GraphConvEmbeddedGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.gcl1 = GCLayer(output_dim, output_dim)
        self.gcl2 = GCLayer(output_dim, output_dim)

        self.register_parameter(
            "w_a", nn.Parameter(torch.randn(output_dim, output_dim).detach())
        )

        self.enc_obs = Encoder(input_dim, output_dim, h_dims=[64], last_act=True)

    def forward(self, data):
        enc = self.enc_obs(data)
        A = torch.matmul(torch.matmul(enc, self.w_a), enc.permute(0, 2, 1))
        adj_mat = softmax(A, dim=2)
        obs_gc = self.gcl1(enc, adj_mat) + enc
        out = self.gcl2(obs_gc, adj_mat) + obs_gc

        return out


class EGGC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.gcl = GCLayer(output_dim, output_dim)

        self.register_parameter(
            "w_a", nn.Parameter(torch.randn(output_dim, output_dim).detach())
        )
        self.encorder = Encoder(input_dim, output_dim, h_dims=[64], last_act=True)

    def forward(self, data):
        enc = self.encorder(data)
        A = torch.matmul(torch.matmul(enc, self.w_a), enc.permute(0, 2, 1))
        adj_mat = softmax(A, dim=2)
        out = self.gcl(enc, adj_mat)

        return out


class GAT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        concat=True,
        n_heads=8,
        dropout_rate=0.0,
        alpha=0.2,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        if concat:
            assert output_dim % n_heads == 0
            self.h_dim = output_dim // n_heads
        else:
            self.h_dim = output_dim

        self.output_dim = output_dim

        self.gat = GraphAttentionLayer(
            in_features=output_dim,
            out_features=output_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )
        
        

    def forward(self, data):
        out = self.gat(data)
        return out