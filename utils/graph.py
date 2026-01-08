import torch
import networkx as nx
import numpy as np
import math


def pos_to_graph(pos, norm_lap_matr=True, th_range=None):
    stack_size = pos.shape[0]
    # obs = pos.reshape((stack_size, int(pos.shape[1]/4), 4))
    max_nodes = pos.shape[1]

    V = torch.as_tensor(pos[:, :, 0:2], dtype=torch.float32)
    V_diff = V.repeat(1, max_nodes, 1) - torch.repeat_interleave(V, max_nodes, dim=1)
    V_norm = torch.linalg.norm(V_diff, dim=2).view(stack_size, max_nodes, max_nodes)
    if th_range != None:
        V_norm[V_norm < th_range] = 1
        V_norm[V_norm >= th_range] = 0

    adj_mat = 1 / (V_norm + torch.eye(max_nodes).to(V_norm.device))
    adj_mat[torch.isinf(adj_mat)] = 0
    if norm_lap_matr:
        d = torch.diag_embed(torch.sum(adj_mat, axis=1))
        delta = torch.diag_embed(1 / torch.sqrt(torch.sum(adj_mat, axis=1)))

        A = torch.bmm(delta, d - adj_mat)
        A = torch.bmm(A, delta)

    else:
        A = adj_mat

    return V, A


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)
