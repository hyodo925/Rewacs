# -*- coding: utf-8 -*-

import numpy as np
import torch


class Trajectory:
    # def __init__(self, obs=None, act=None, r_obs=None, next_obs=None, next_r_obs=None, obs0=None, r_obs0=None, rwd=None, v=None):
    def __init__(
        self, obs=None, act=None, r_obs=None, obs0=None, r_obs0=None, rwd=None, v=None
    ):
        self.obs = obs
        self.act = act
        self.r_obs = r_obs
        self.obs0 = obs0
        self.r_obs0 = r_obs0
        # self.next_obs = obs
        # self.next_r_obs = r_obs
        self.rwd = rwd
        self.v = v

    @property
    def length(self):
        return self.obs.shape[0]
