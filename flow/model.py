import torch
import torch.nn as nn
import numpy as np
from flow.flows import RealNVP, GrevNet
from flow.graph_models import Encoder
import matplotlib.pyplot as plt
import seaborn as sns
class SituationFlow(nn.Module):
    def __init__(
        self,
        input_dim,
        n_flow_blocks,
        n_flow_hidden_num,
        h_dim,
        debug=False,
        device="cpu",
    ):
        super(SituationFlow, self).__init__()

        self.projection_dim = h_dim
        self.debug = debug
        self.device = device

        self.flow = RealNVP(
            n_blocks=n_flow_blocks,
            n_hidden=n_flow_hidden_num,
            input_dim=input_dim,
            h_dim=h_dim,
        )
        self.register_buffer("theta", torch.zeros([]))

    def forward(self, data):
        z, log_det_j = self.flow(data)

        return z, log_det_j

    def get_switching_score(self, data):
        z, _ = self.forward(data.to(self.device))
        switching_score = torch.mean(z**2)
        return switching_score

    def set_switching_threshold(self, data):
        with torch.no_grad():
            switching_score = self.get_switching_score(data)
        self.register_buffer("theta", switching_score)
        print(
            "The switching threshold is set to {}".format(switching_score.data.item())
        )

    def switching_necessity(self, data):
        switching_score = self.get_switching_score(data)
        if switching_score.data.item() > self.theta:
            return True
        else:
            return False

    def to(self, device):
        super().to(device)
        self.device = device

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class GraphSituationFlow(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_flow_blocks,
        n_flow_hidden_num,
        h_dim,
        enc_hdims=[64],
        threshold_type="mean",
        device="cpu",
    ):
        super(GraphSituationFlow, self).__init__()

        self.projection_dim = h_dim
        self.device = device
        self.threshold_type = threshold_type
        self.h_dim = h_dim
        self.switching_score_each_ped = 0
        self.encoder = Encoder(obs_dim, h_dim, h_dims=enc_hdims, last_act=True)
        self.flow = GrevNet(
            n_blocks=n_flow_blocks,
            n_hidden=n_flow_hidden_num,
            input_dim=h_dim,
            h_dim=h_dim,
        )

        self.register_buffer("theta", torch.zeros([]))

    def forward(self, data):
        encorded = self.encord(data.to(self.device))
        z, log_det_j = self.flow.inverse(encorded)
        # z, log_det_j = self.flow(data.to(self.device))

        return z, log_det_j

    def encord(self, data):
        encoded = self.encoder(data)
        return encoded

    def get_switching_score(self, data):
        z, _ = self.forward(data)
        if self.threshold_type == "mean":
            switching_score = torch.mean(z**2)
            # switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))) / (z.shape[1])
        elif self.threshold_type == "sum":
            switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,)))
        elif self.threshold_type == "max":
            switching_score = torch.max(
                torch.mean(torch.mean(z**2, dim=1), dim=1)
            )
        return switching_score

    def get_switching_score_dim(self, data):
        z, _ = self.forward(data)
        if self.threshold_type == "mean":
            dims = tuple(range(1, z.ndim))
            switching_score = torch.mean(z**2, dim=dims)
            # switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))) / (z.shape[1])
        elif self.threshold_type == "sum":
            dims = tuple(range(1, z.ndim))
            switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,)), dim=dims)
        elif self.threshold_type == "max":
            dims = tuple(range(1, z.ndim))
            switching_score = torch.max(
                torch.mean(torch.mean(z**2, dim=1), dim=dims)
            )
        return switching_score
    
    def set_switching_threshold(self, data):
        with torch.no_grad():
            # anomaly_score = self.get_anomaly_score(data)
            z, _ = self.forward(data)
            if self.threshold_type == "mean":
                switching_score = torch.mean(z**2)
                # switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))) / (z.shape[1])
            elif self.threshold_type == "sum":
                switching_score = torch.mean(torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,)))
            elif self.threshold_type == "max":
                switching_score = torch.max(
                    torch.mean(torch.mean(z**2, dim=1), dim=1)
                )
        self.register_buffer("theta", switching_score)
        print(
            "The swithcing threshold is set to {}".format(switching_score.data.item())
        )
        return switching_score

    def switching_necessity(self, data):
        switching_score = self.get_switching_score(data)
        if switching_score.data.item() > self.theta:
            return True
        else:
            return False
        
    def flow_loss(self, z, log_det_j, graph=False):
        if graph:
            n_step = log_det_j.shape[1]
            if self.threshold_type == "mean":
                return torch.mean(
                    torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))
                    - torch.sum(log_det_j, dim=(1,))
                ) / (z.shape[1])
            elif self.threshold_type == "sum":
                return torch.mean( torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,)) - torch.sum(log_det_j, dim=(1,)) )
        else:
            return torch.mean(0.5 * torch.sum(z**2, dim=(1,)) - log_det_j) / (z.shape[1])
    
    def plot_log_prob_comparison(self, model, train_loader, ood_loader,  device, save_path=None):
        model.eval()
        train_scores = []
        ood_scores = []
        with torch.no_grad():
            for i in range(100):
                data = train_loader.sample(100)["humans_obs"].to(device)
                scores = model.get_switching_score(data)
                train_scores.append(scores.item())

            for i in range(100):
                data = ood_loader.sample(100)["humans_obs"].to(device)
                scores = model.get_switching_score(data)
                ood_scores.append(scores.item())

        plt.figure(figsize=(8, 5))
        sns.set_style("darkgrid")

        sns.histplot(train_scores, color="gray", label="Train (ID)", kde=True, stat="density", alpha=0.6)
        sns.histplot(ood_scores, color="crimson", label="Test (OOD)", kde=True, stat="density", alpha=0.6)
        plt.axvline(self.theta.item(),color="blue",linestyle="--",linewidth=2,label=f"Switching threshold = {self.theta.item():.4f}")
        plt.xlabel("Switching score τ", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.title("Log-Probability Distribution Comparison", fontsize=15)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


    def to(self, device):
        super().to(device)
        self.device = device

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # print(torch.load(path,map_location=torch.device('cpu')).keys())
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

        
