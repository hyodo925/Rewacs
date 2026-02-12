import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torch
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


def trajectory_collate(Trajectory):
    return Trajectory


class TrajectoryBuffer:
    def __init__(self, capacity=5e3):
        self.trajs = []
        self.size = 0
        self.capacity = int(capacity)

    def add(self, traj):
        if (self.size + len(traj)) > self.capacity:
            diff = int((self.size + len(traj)) - self.capacity)
            del self.trajs[0:diff]
            self.trajs += traj
        else:
            self.trajs += traj
            self.size += len(traj)

    def set(self, trajs):
        self.trajs = trajs
        self.size = len(trajs)

    def clear(self):
        del self.trajs[:]
        self.size = 0

    def fetch(self, batch_size=20):
        # ind = np.random.choice(np.arange(self.size), batch_size, False)
        # traj_samples = []
        # for i in range(batch_size):
        #     traj_samples.append(self.trajs[1])
        traj_samples = random.sample(self.trajs, batch_size)
        # traj_samples = self.trajs[0:2]
        return traj_samples

    def loader(self, batch_siz=5):
        return list(chunked(random.sample(self.trajs, len(self.trajs)), batch_siz))


class ClassifiedTrajectoryBuffer:
    def __init__(self, capacity=6e3):
        self.trajs_model = []
        self.trajs_success = []
        self.trajs_collision = []
        self.size_model = 0
        self.size_success = 0
        self.size_collision = 0
        self.capacity = int(capacity)
        self.capa_model = int(self.capacity / 3)
        self.capa_success = int(self.capacity / 3)
        self.capa_collision = int(self.capacity / 3)

    def add_success(self, traj):
        if (self.size_success + len(traj)) > self.capa_success:
            diff = int((self.size_success + len(traj)) - self.capa_success)
            del self.trajs_success[0:diff]
            self.trajs_success += traj
        else:
            self.trajs_success += traj
            self.size_success += len(traj)

    def add_collision(self, traj):
        if (self.size_collision + len(traj)) > self.capa_collision:
            diff = int((self.size_collision + len(traj)) - self.capa_collision)
            del self.trajs_collision[0:diff]
            self.trajs_collision += traj
        else:
            self.trajs_collision += traj
            self.size_collision += len(traj)

    def set_model_trajs(self, trajs):
        self.trajs_model = trajs
        self.size_model = len(trajs)

    def clear(self):
        del self.trajs_model[:]
        del self.trajs_success[:]
        del self.trajs_collision[:]
        self.size_model = 0
        self.size_success = 0
        self.size_collision = 0

    def fetch(self, batch_size_model=5, batch_size_success=5, batch_size_collision=5):
        # ind = np.random.choice(np.arange(self.size), batch_size, False)
        if (
            self.size_success > batch_size_success
            and self.size_collision > batch_size_collision
        ):
            traj_samples_model = random.sample(self.trajs_model, batch_size_model)
            traj_samples_success = random.sample(self.trajs_success, batch_size_success)
            traj_samples_collision = random.sample(
                self.trajs_collision, batch_size_collision
            )
            traj_samples = (
                traj_samples_model + traj_samples_success + traj_samples_collision
            )
        else:
            traj_samples = random.sample(
                self.trajs_model,
                batch_size_model + batch_size_success + batch_size_collision,
            )

        return traj_samples


class OffPolicyReplayBuffer:
    def __init__(self, capacity=1e6):
        self.trajs = []
        self.size = 0
        self.capacity = int(capacity)

    def add(self, traj):
        if (self.size + len(traj)) > self.capacity:
            diff = int((self.size + len(traj)) - self.capacity)
            del self.trajs[0:diff]
            self.trajs += traj
        else:
            self.trajs += traj
            self.size += len(traj)

    def set(self, trajs):
        self.trajs = trajs
        self.size = len(trajs)

    def clear(self):
        del self.trajs[:]
        self.size = 0

    def fetch(self, batch_size=100):
        # ind = np.random.choice(np.arange(self.size), batch_size, False)
        traj_samples = random.sample(self.trajs, batch_size)

        return traj_samples


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

    # def set_data(self, data):
    #     self.memory = data


class ODPRSampler(PrioritizedSampler):
    def __init__(self, capacity, alpha=1.0, beta=0.4):
        # ロジック側で重みを調整済みのため、alpha=1.0 を推奨（重みをそのまま確率にするため）
        super().__init__(max_capacity=capacity, alpha=alpha, beta=beta)

    def apply_odpr_weights(self, weights, memory_size):
        epsilon = 1e-37
        weights = torch.where(torch.abs(weights) < epsilon, 
                             torch.tensor(0.0, dtype=torch.float64), 
                             weights)
        
        std = torch.std(weights, unbiased=False)
        scale = 2 / std if std > 0 else 1.0
        
        if scale > 1:
            weights = torch.clamp(scale * (weights - 1) + 1, min=0.1)
            weights = weights / (torch.nansum(weights) + epsilon) * memory_size
        indices = torch.arange(len(weights))
        self.update_priority(indices, weights.float())