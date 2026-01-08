# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class RLNavigation(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        bc_flow,
        device="cpu",
    ):
        super(RLNavigation, self).__init__()

        self.actor = actor
        self.critic = critic
        self.bc_flow = bc_flow
        self.device = device

    def to(self, device):
        self.device = device
        return super().to(device)

    def generate_action(self, state):
        action, log_prob, mean = self.actor.sample(state)
        return action, log_prob, mean

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
