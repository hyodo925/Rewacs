import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.nn.functional import softmax

from utils.graph import pos_to_graph
from utils.layers import GCLayer, GraphAttentionLayer 


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class MLP(nn.Module):
    def __init__(self, D, d, h_dims=[64], last_act=True):
        super().__init__()
        self.net = self.make_mlp([D] + h_dims + [d], last_act=last_act)
        # self.init_weights(init_w)

    def make_mlp(self, mlp_dims, activation="mish", last_act=False):
        layers = []
        mlp_dims = mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_act:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif activation == "mish":
                    layers.append(nn.Mish())
        net = nn.Sequential(*layers)
        return net

    def forward(self, data):
        out = self.net(data)

        return out


class GCObsAggregator(nn.Module):
    def __init__(
        self, obs_dim, r_obs_dim, projection_dim, enc_hdims=[64], prediction=False
    ):
        super().__init__()
        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.enc_obs = MLP(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.gcl1 = GCLayer(projection_dim, projection_dim)
        self.gcl2 = GCLayer(projection_dim, projection_dim)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        # self.register_parameter('w_a', nn.Parameter(torch.randn(projection_dim, projection_dim).detach()))

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        obs_cat = torch.cat(
            (
                r_obs[:, :, :2],
                obs.reshape((n, -1, self.obs_dim))[:, :, :2],
            ),
            1,
        )
        _, adj_mat = pos_to_graph(obs_cat.reshape(-1, p_num + 1, 2).detach())

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )
        # A = torch.matmul(torch.matmul(obs_stack, self.w_a), obs_stack.permute(0, 2, 1))
        # adj_mat = softmax(A, dim=2)
        obs_gc = self.gcl1(obs_stack, adj_mat) + obs_stack
        obs_gc2 = self.gcl2(obs_gc, adj_mat) + obs_gc

        if self.prediction:
            integrated = obs_gc2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gc2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)


class EmbeddedGaussianAggregator(nn.Module):
    def __init__(
        self, obs_dim, r_obs_dim, projection_dim, enc_hdims=[64], prediction=False
    ):
        super().__init__()
        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.enc_obs = MLP(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.gcl1 = GCLayer(projection_dim, projection_dim)
        self.gcl2 = GCLayer(projection_dim, projection_dim)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        self.register_parameter(
            "w_a", nn.Parameter(torch.randn(projection_dim, projection_dim).detach())
        )

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )
        A = torch.matmul(torch.matmul(obs_stack, self.w_a), obs_stack.permute(0, 2, 1))
        adj_mat = softmax(A, dim=2)
        obs_gc = self.gcl1(obs_stack, adj_mat) + obs_stack
        obs_gc2 = self.gcl2(obs_gc, adj_mat) + obs_gc
        if self.prediction:
            integrated = obs_gc2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gc2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)


class GATAggregator(nn.Module):
    def __init__(
        self,
        obs_dim,
        r_obs_dim,
        projection_dim,
        enc_hdims=[64],
        concat=True,
        n_heads=8,
        prediction=False,
        dropout_rate=0.0,
        alpha=0.2,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        if concat:
            assert projection_dim % n_heads == 0
            self.h_dim = projection_dim // n_heads
        else:
            self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.enc_obs = MLP(obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)
        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim

        self.gat1 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat2 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        p_num = obs.shape[1]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        obs_stack = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        obs_gat1 = self.gat1(obs_stack) + obs_stack
        obs_gat2 = self.gat2(obs_gat1) + obs_gat1
        if self.prediction:
            integrated = obs_gat2[:, 1:, :].reshape(-1, p_num, self.output_dim)
        else:
            integrated = obs_gat2[:, 0, :].reshape(-1, self.output_dim)

        return integrated
        # return obs_stack.reshape(n, -1)


class MAGATAggregator(nn.Module):
    def __init__(
        self,
        r_obs_dim,
        o_r_obs_dim,
        h_obs_dim,
        projection_dim,
        enc_hdims=[64],
        concat=True,
        n_heads=8,
        prediction=False,
        dropout_rate=0.0,
        alpha=0.2,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        if concat:
            assert projection_dim % n_heads == 0
            self.h_dim = projection_dim // n_heads
        else:
            self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.enc_o_r_obs = MLP(
            o_r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )

        self.enc_h_obs = MLP(h_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.output_dim = projection_dim
        self.prediction = prediction

        self.projection_dim = projection_dim
        self.r_obs_dim = r_obs_dim
        self.h_obs_dim = h_obs_dim

        # self.gat_r_1 = MAGraphAttentionLayer(
        #     in_features=projection_dim,
        #     out_features=projection_dim,
        #     n_heads=n_heads,
        #     dropout_rate=dropout_rate,
        #     alpha=alpha,
        #     concat=concat,
        # )

        # self.gat_r_2 = MAGraphAttentionLayer(
        #     in_features=projection_dim,
        #     out_features=projection_dim,
        #     n_heads=n_heads,
        #     dropout_rate=dropout_rate,
        #     alpha=alpha,
        #     concat=concat,
        # )

        # self.gat_h_1 = MAGraphAttentionLayer(
        #     in_features=projection_dim,
        #     out_features=projection_dim,
        #     n_heads=n_heads,
        #     dropout_rate=dropout_rate,
        #     alpha=alpha,
        #     concat=concat,
        # )

        # self.gat_h_2 = MAGraphAttentionLayer(
        #     in_features=projection_dim,
        #     out_features=projection_dim,
        #     n_heads=n_heads,
        #     dropout_rate=dropout_rate,
        #     alpha=alpha,
        #     concat=concat,
        # )
        self.gat_r_1 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat_r_2 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat_h_1 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        self.gat_h_2 = GraphAttentionLayer(
            in_features=projection_dim,
            out_features=projection_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            alpha=alpha,
            concat=concat,
        )

        # self.gat_l = GraphAttentionLayer(
        #     in_features=projection_dim * 2,
        #     out_features=projection_dim * 2,
        #     n_heads=n_heads,
        #     dropout_rate=dropout_rate,
        #     alpha=alpha,
        #     concat=concat,
        # )
        # self.multihead_attn = nn.MultiheadAttention(
        #     projection_dim * 2, num_heads=n_heads, batch_first=True
        # )

    def propagate_gnn(self, r_obs, obs, enc_obs_func, gat1, gat2):
        n = obs.shape[0]
        # agent_num = obs.shape[1]
        others_num = obs.shape[2]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.r_obs_dim))

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = enc_obs_func(obs)
        obs_stack = torch.cat((enc_r_obs, enc_obs), 1)

        obs_gat1 = gat1(obs_stack) + obs_stack
        obs_gat2 = gat2(obs_gat1) + obs_gat1
        # if self.prediction:
        #     integrated = obs_gat2[:, :, 1:, :].reshape(
        #         n, agent_num, others_num, self.output_dim
        #     )
        # else:
        #     integrated = obs_gat2[:, :, 0, :].reshape(n, agent_num, self.output_dim)
        if self.prediction:
            aggr = obs_gat2[:, 1:, :].reshape(n, others_num, self.output_dim)
        else:
            aggr = obs_gat2[:, 0, :].reshape(n, self.output_dim)

        # return enc_r_obs.reshape(n, agent_num, self.output_dim)
        return aggr

    def forward(self, r_obs, o_r_obs, h_obs):
        aggr_r = self.propagate_gnn(
            r_obs, o_r_obs, self.enc_o_r_obs, self.gat_r_1, self.gat_r_2
        )
        aggr_h = self.propagate_gnn(
            r_obs, h_obs, self.enc_h_obs, self.gat_h_1, self.gat_h_2
        )

        # aggr = torch.cat([aggr_r, aggr_h], dim=-1)
        aggr = aggr_r + aggr_h

        # integrated = self.gat_l(integrated)
        # integrated, _ = self.multihead_attn(integrated, integrated, integrated)

        return aggr
        # return obs_stack.reshape(n, -1)


class MASelfAttentionAggregator_(nn.Module):
    def __init__(
        self,
        r_obs_dim,
        o_r_obs_dim,
        h_obs_dim,
        projection_dim,
        enc_hdims=[64],
        concat=True,
        n_heads=8,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        if concat:
            assert projection_dim % n_heads == 0
            self.h_dim = projection_dim // n_heads
        else:
            self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.enc_o_r_obs = MLP(
            o_r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )

        self.enc_h_obs = MLP(h_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.output_dim = projection_dim

        self.projection_dim = projection_dim
        self.r_obs_dim = r_obs_dim
        self.h_obs_dim = h_obs_dim

        self.multihead_attn_r = nn.MultiheadAttention(
            projection_dim, num_heads=n_heads, batch_first=True
        )

        self.multihead_attn_h = nn.MultiheadAttention(
            projection_dim, num_heads=n_heads, batch_first=True
        )

        # self.multihead_attn_cross_modal = nn.MultiheadAttention(
        #     projection_dim, num_heads=n_heads, batch_first=True
        # )

        self.multihead_attn_last = nn.MultiheadAttention(
            projection_dim, num_heads=n_heads, batch_first=True
        )

        # self.enc_out_attn_h = MLP(
        #     projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        # )
        # self.enc_out_attn_r = MLP(
        #     projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        # )

        # self.enc_cross_attn_qv = MLP(
        #     projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        # )

    # def forward(self, r_obs, o_r_obs, h_obs):
    #     enc_r_obs = self.enc_r_obs(r_obs)
    #     enc_o_r_obs = self.enc_o_r_obs(o_r_obs)
    #     enc_h_obs = self.enc_h_obs(h_obs)
    #     obs_stack_r = torch.cat((enc_r_obs, enc_o_r_obs), 2)
    #     obs_stack_h = torch.cat((enc_r_obs, enc_h_obs), 2)

    #     B_r, N_r, M_r, D_r = obs_stack_r.shape
    #     obs_stack_r_ = obs_stack_r.reshape(B_r * N_r, M_r, D_r)

    #     B_h, N_h, M_h, D_h = obs_stack_h.shape
    #     obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)

    #     out_r, _ = self.multihead_attn_r(obs_stack_r_, obs_stack_r_, obs_stack_r_)
    #     int_r = out_r.reshape(B_r, N_r, M_r, D_r)[:, :, 0, :]

    #     out_h, _ = self.multihead_attn_h(obs_stack_h_, obs_stack_h_, obs_stack_h_)
    #     int_h = out_h.reshape(B_h, N_h, M_h, D_h)[:, :, 0, :]

    #     integrated = torch.cat([int_r, int_h], dim=-1)

    #     # integrated = self.gat_l(integrated)
    #     # integrated_, _ = self.multihead_attn_last(integrated, integrated, integrated)

    #     return integrated
    # return obs_stack.reshape(n, -1)

    def forward(self, r_obs, o_r_obs, h_obs):
        enc_r_obs = self.enc_r_obs(r_obs)
        enc_h_obs = self.enc_h_obs(h_obs)
        obs_stack_h = torch.cat((enc_r_obs, enc_h_obs), 1)

        # B_r, M_r, D_r = obs_stack_r.shape
        # obs_stack_r_ = obs_stack_r.reshape(B_r, M_r, D_r)

        # B_h, N_h, M_h, D_h = obs_stack_h.shape
        # obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)
        aggr = enc_r_obs
        out_h, _ = self.multihead_attn_h(obs_stack_h, obs_stack_h, obs_stack_h)
        # out_h = self.enc_out_attn_h(out_h)
        aggr = aggr + out_h

        if o_r_obs.shape[1] > 0:
            enc_o_r_obs = self.enc_o_r_obs(o_r_obs)

            obs_stack_r = torch.cat((enc_r_obs, enc_o_r_obs), 1)

            out_r, _ = self.multihead_attn_r(obs_stack_r, obs_stack_r, obs_stack_r)
            # out_r = self.enc_out_attn_r(out_r)

            aggr = aggr + out_r

        return aggr[:, 0, :]

    # def forward(self, r_obs, o_r_obs, h_obs):
    #     enc_r_obs = self.enc_r_obs(r_obs)
    #     enc_h_obs = self.enc_h_obs(h_obs)
    #     obs_stack_h = torch.cat((enc_r_obs, enc_h_obs), 1)

    #     # B_r, M_r, D_r = obs_stack_r.shape
    #     # obs_stack_r_ = obs_stack_r.reshape(B_r, M_r, D_r)

    #     # B_h, N_h, M_h, D_h = obs_stack_h.shape
    #     # obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)

    #     out_h, _ = self.multihead_attn_h(obs_stack_h, obs_stack_h, obs_stack_h)
    #     int_h = out_h[:, 0, :].unsqueeze(1)
    #     # integrated = out_h[:, 0, :]

    #     if o_r_obs.shape[1] > 0:
    #         enc_o_r_obs = self.enc_o_r_obs(o_r_obs)

    #         obs_stack_r = torch.cat((enc_r_obs, enc_o_r_obs), 1)

    #         out_r, _ = self.multihead_attn_r(obs_stack_r, obs_stack_r, obs_stack_r)

    #         int_r = out_r[:, 0, :].unsqueeze(1)
    #         # integrated += out_r[:, 0, :]

    #         int_fused = torch.cat([int_h, int_r], dim=1)

    #         integrated_h, _ = self.multihead_attn_last(int_h, int_fused, int_fused)
    #         integrated_r, _ = self.multihead_attn_last(int_r, int_fused, int_fused)
    #         integrated = torch.cat([integrated_h, integrated_r], dim=1).mean(dim=1)

    #     else:
    #         int_fused = out_h
    #         integrated_, _ = self.multihead_attn_last(int_h, int_fused, int_fused)
    #         integrated = integrated_[:, 0, :]
    #     # integrated = torch.cat([int_r, int_h], dim=-1)

    #     # integrated = self.gat_l(integrated)
    #     # integrated, _ = self.multihead_attn_last(enc_r_obs, int_fused, int_fused)

    #     # return integrated[:, 0, :]
    #     return integrated

    # def forward(self, r_obs, o_r_obs, h_obs):
    #     enc_r_obs = self.enc_r_obs(r_obs)
    #     enc_h_obs = self.enc_h_obs(h_obs)
    #     # obs_stack_h = torch.cat((enc_r_obs, enc_h_obs), 1)

    #     # B_r, M_r, D_r = obs_stack_r.shape
    #     # obs_stack_r_ = obs_stack_r.reshape(B_r, M_r, D_r)

    #     # B_h, N_h, M_h, D_h = obs_stack_h.shape
    #     # obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)

    #     # aggr = enc_r_obs[:, 0, :]

    #     attn_rh, _ = self.multihead_attn_h(
    #         enc_r_obs,
    #         enc_h_obs,
    #         enc_h_obs,
    #     )
    #     attn_rh = self.enc_out_attn_h(attn_rh)
    #     # int_h = out_h[:, 0, :].unsqueeze(1)
    #     # integrated = out_h[:, 0, :]

    #     if o_r_obs.shape[1] > 0:
    #         enc_o_r_obs = self.enc_o_r_obs(o_r_obs)

    #         # obs_stack_r = torch.cat((enc_r_obs, enc_o_r_obs), 1)

    #         attn_rr, _ = self.multihead_attn_r(
    #             enc_r_obs,
    #             enc_o_r_obs,
    #             enc_o_r_obs,
    #         )

    #         attn_rr = self.enc_out_attn_r(attn_rr)

    #         # aggr = enc_r_obs + attn_rh + attn_rr

    #         # int_r = out_r[:, 0, :].unsqueeze(1)
    #         # integrated += out_r[:, 0, :]

    #         attn = torch.cat([enc_r_obs, attn_rh, attn_rr], dim=1)

    #     else:
    #         # aggr = enc_r_obs + attn_rh
    #         attn = torch.cat([enc_r_obs, attn_rh], dim=1)

    #     # integrated = torch.cat([int_r, int_h], dim=-1)
    #     attn_enc = self.enc_cross_attn_qv(attn)

    #     # integrated = attn_stack
    #     attn_cross, _ = self.multihead_attn_cross_modal(attn, attn_enc, attn_enc)

    #     aggr, _ = self.multihead_attn_last(attn_cross, attn_cross, attn_cross)
    #     # aggr = torch.cat([enc_r_obs, enc_r_obs], dim=-1)

    #     # return aggr[:, 0, :]
    #     return aggr.mean(dim=1)


class MASelfAttentionAggregator(nn.Module):
    def __init__(
        self,
        r_obs_dim,
        o_r_obs_dim,
        h_obs_dim,
        projection_dim,
        enc_hdims=[64],
        pooling=False,
        n_heads=8,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.h_dim = projection_dim

        self.projection_dim = projection_dim

        self.enc_r_obs = MLP(r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.enc_o_r_obs = MLP(
            o_r_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )

        self.enc_h_obs = MLP(h_obs_dim, projection_dim, h_dims=enc_hdims, last_act=True)

        self.enc_out_attn_h = MLP(
            projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )
        self.enc_out_attn_r = MLP(
            projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        )

        # self.enc_cross_attn_qv = MLP(
        #     projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        # )
        # self.enc_qv = MLP(
        #     projection_dim, projection_dim, h_dims=enc_hdims, last_act=True
        # )

        self.pooling = pooling

        self.output_dim = projection_dim

        self.projection_dim = projection_dim
        self.r_obs_dim = r_obs_dim
        self.h_obs_dim = h_obs_dim

        # self.self_attn_r = nn.MultiheadAttention(
        #     projection_dim, num_heads=n_heads, batch_first=True
        # )

        # self.self_attn_h = nn.MultiheadAttention(
        #     projection_dim, num_heads=n_heads, batch_first=True
        # )

        self.multihead_attn_r = nn.MultiheadAttention(
            projection_dim, num_heads=n_heads, batch_first=True
        )

        self.multihead_attn_h = nn.MultiheadAttention(
            projection_dim, num_heads=n_heads, batch_first=True
        )

        # self.multihead_attn_cross_modal = nn.MultiheadAttention(
        #     projection_dim, num_heads=n_heads, batch_first=True
        # )

        # self.multihead_attn_last = nn.MultiheadAttention(
        #     projection_dim, num_heads=n_heads, batch_first=True
        # )

    def forward(self, r_obs, o_r_obs, h_obs):
        # out_list = []
        # enc_r_obs = self.enc_r_obs(r_obs)
        # enc_h_obs = self.enc_h_obs(h_obs)
        # enc_o_r_obs = self.enc_o_r_obs(o_r_obs)
        # for i in range(r_obs.shape[1]):
        #     # B_r, M_r, D_r = obs_stack_r.shape
        #     # obs_stack_r_ = obs_stack_r.reshape(B_r, M_r, D_r)

        #     # B_h, N_h, M_h, D_h = obs_stack_h.shape
        #     # obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)
        #     aggr = enc_r_obs[:, i, :, :]
        #     out_h, _ = self.multihead_attn_h(
        #         enc_r_obs[:, i, :, :], enc_h_obs[:, i, :, :], enc_h_obs[:, i, :, :]
        #     )
        #     aggr = aggr + out_h

        #     if o_r_obs.shape[1] > 0:
        #         out_r, _ = self.multihead_attn_r(
        #             enc_r_obs[:, i, :, :],
        #             enc_o_r_obs[:, i, :, :],
        #             enc_o_r_obs[:, i, :, :],
        #         )

        #         aggr = aggr + out_r

        #     out_list.append(aggr)

        # out_stack = torch.stack(out_list, dim=1)

        # if self.pooling:
        #     out_stack = out_stack.mean(dim=1, keepdim=True)

        # return out_stack[:, :, 0, :]

        bs, an, _, _ = r_obs.shape
        # out_list = []
        enc_r_obs = self.enc_r_obs(r_obs).reshape(bs * an, -1, self.projection_dim)
        aggr = enc_r_obs

        if h_obs.shape[2] > 0:
            enc_h_obs = self.enc_h_obs(h_obs).reshape(bs * an, -1, self.projection_dim)

            # sa_enc_h_obs, _ = self.self_attn_h(enc_h_obs, enc_h_obs, enc_h_obs)
            out_h, _ = self.multihead_attn_h(enc_r_obs, enc_h_obs, enc_h_obs)
            out_h = self.enc_out_attn_h(out_h)
            aggr = aggr + out_h

        if o_r_obs.shape[2] > 0:
            enc_o_r_obs = self.enc_o_r_obs(o_r_obs).reshape(
                bs * an, -1, self.projection_dim
            )
            # sa_enc_o_r_obs, _ = self.self_attn_r(enc_o_r_obs, enc_o_r_obs, enc_o_r_obs)
            out_r, _ = self.multihead_attn_r(
                enc_r_obs,
                enc_o_r_obs,
                enc_o_r_obs,
            )
            out_r = self.enc_out_attn_r(out_r)

            aggr = aggr + out_r

        output = aggr.reshape(bs, an, self.projection_dim)

        if self.pooling:
            output = output.mean(dim=1, keepdim=True)

        return output

    # def forward(self, r_obs, o_r_obs, h_obs):
    #     bs, an, _, _ = r_obs.shape
    #     enc_r_obs = self.enc_r_obs(r_obs).reshape(bs * an, -1, self.projection_dim)
    #     # obs_stack_h = torch.cat((enc_r_obs, enc_h_obs), 1)

    #     # B_r, M_r, D_r = obs_stack_r.shape
    #     # obs_stack_r_ = obs_stack_r.reshape(B_r, M_r, D_r)

    #     # B_h, N_h, M_h, D_h = obs_stack_h.shape
    #     # obs_stack_h_ = obs_stack_h.reshape(B_h * N_h, M_h, D_h)

    #     # aggr = enc_r_obs[:, 0, :]

    #     if h_obs.shape[2] > 0:
    #         enc_h_obs = self.enc_h_obs(h_obs).reshape(bs * an, -1, self.projection_dim)

    #         attn_rh, _ = self.multihead_attn_h(
    #             enc_r_obs,
    #             enc_h_obs,
    #             enc_h_obs,
    #         )
    #         attn_rh = self.enc_out_attn_h(attn_rh)
    #         # int_h = out_h[:, 0, :].unsqueeze(1)
    #         # integrated = out_h[:, 0, :]

    #     if o_r_obs.shape[1] > 0:
    #         enc_o_r_obs = self.enc_o_r_obs(o_r_obs).reshape(
    #             bs * an, -1, self.projection_dim
    #         )

    #         # obs_stack_r = torch.cat((enc_r_obs, enc_o_r_obs), 1)

    #         attn_rr, _ = self.multihead_attn_r(
    #             enc_r_obs,
    #             enc_o_r_obs,
    #             enc_o_r_obs,
    #         )

    #         attn_rr = self.enc_out_attn_r(attn_rr)

    #         # aggr = enc_r_obs + attn_rh + attn_rr

    #         # int_r = out_r[:, 0, :].unsqueeze(1)
    #         # integrated += out_r[:, 0, :]

    #         attn = torch.cat([enc_r_obs, attn_rh, attn_rr], dim=1)

    #     else:
    #         # aggr = enc_r_obs + attn_rh
    #         attn = torch.cat([enc_r_obs, attn_rh], dim=1)

    #     # integrated = torch.cat([int_r, int_h], dim=-1)
    #     attn_enc = self.enc_cross_attn_qv(attn)

    #     # integrated = attn_stack
    #     attn_cross, _ = self.multihead_attn_cross_modal(attn, attn_enc, attn_enc)

    #     aggr, _ = self.multihead_attn_last(attn_cross, attn_cross, attn_cross)
    #     # aggr = torch.cat([enc_r_obs, enc_r_obs], dim=-1)

    #     # return aggr[:, 0, :]
    #     return aggr.mean(dim=1).reshape(bs, an, self.projection_dim)


class SimpleConcatenateObsAggregator(nn.Module):
    def __init__(self, obs_dim, r_obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim

    def forward(self, obs, r_obs):
        n = obs.shape[0]
        if len(r_obs.shape) < 3:
            r_obs = r_obs.reshape((n, 1, self.obs_dim))

        aggr = torch.cat((r_obs.reshape((n, -1)), obs.reshape((n, -1))), 1)

        return aggr

    # def forward(self, state, obs, r_obs):

    #     enc_r_obs = self.enc_r_obs(r_obs)
    #     return enc_r_obs


class ConcatenateObsAggregator(nn.Module):
    def __init__(self, obs_dim, r_obs_dim, projection_dim):
        super().__init__()
        self.enc_r_obs = MLP(r_obs_dim, projection_dim, last_relu=True)
        self.enc_obs = MLP(obs_dim, projection_dim, last_relu=True)
        self.output_dim = projection_dim

        self.projection_dim = projection_dim
        self.obs_dim = obs_dim
        self.r_obs_dim = r_obs_dim
        # self.register_parameter('w_a', nn.Parameter(torch.randn(projection_dim, projection_dim).detach()))

    def forward(self, obs, r_obs):
        n = obs.shape[0]

        p_num = obs.shape[1]

        enc_r_obs = self.enc_r_obs(r_obs)
        enc_obs = self.enc_obs(obs.reshape((n, -1, self.obs_dim)))
        integrated = torch.cat(
            (enc_r_obs.reshape(n, -1, self.projection_dim), enc_obs), 1
        )

        return integrated