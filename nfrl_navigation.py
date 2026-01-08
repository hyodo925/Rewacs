# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class NFRLNavigation(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        device="cpu",
    ):
        super(NFRLNavigation, self).__init__()

        self.actor = actor
        self.critic = critic
        self.device = device

    def to(self, device):
        self.device = device
        return super().to(device)

    def generate_action(self, state, sample_size=1):
        prior_sample = self.actor.prior.sample((sample_size,))
        action = self.actor.reverse(prior_sample, state)
        return action, action, action

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))