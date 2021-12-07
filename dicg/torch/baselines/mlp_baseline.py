import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from garage.torch.algos import filter_valids

class CentValueFunction(nn.Module):
    def __init__(self,
                 state_size,
                 hidden_size=256,
                 device='cpu'):
        
        super().__init__()

        self.device = device

        self.vf = self.vf = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def compute_loss(self, state, returns, valids):
        # returns.shape = (n_paths, max_t)
        est_returns = self.forward(state)
        valid_est_returns = torch.cat(filter_valids(est_returns, valids))
        # flatten len = valids[0] + valids[1] + ...
        # print('valid_est_returns.shape =', valid_est_returns.shape)
        valid_returns = torch.cat(filter_valids(returns, valids))
        return F.mse_loss(valid_est_returns, valid_returns)


    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state).float().to(self.device) # (n_paths, max_t, state_dim)
        est_returns = self.vf(state).squeeze()
        return est_returns