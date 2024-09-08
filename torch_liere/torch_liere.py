import torch
import torch.nn as nn


class LieRE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.entries = nn.Parameter(torch.randn((d*(d-1)//2, 1)) * 0.02, requires_grad=True)  # (D*(D-1)//2,)
        # arrange into skew-symmetric matrix

    def _get_R(self, t):
        A = torch.zeros(self.d, self.d, device=self.entries.device)
        A[torch.tril(torch.ones(self.d, self.d, device=self.entries.device), -1) == 1] = self.entries.view(-1)  # (D, D)
        self.A = A - A.T  # (D, D)
        At = torch.einsum("de,t->tde", self.A, t)
        R = torch.matrix_exp(At)  # (T, D, D)
        return R  # (T, D, D)

    def forward(self, x, offset=0):
        t = torch.arange(x.shape[2]).float().to(x.device)  # (T)
        R = self._get_R(t - offset)  # (T, D, D)
        x_rot = torch.einsum("bhtd,tde->bhte", x, R)  # (B, H, T, D) @ (T, D, D) -> (B, H, D, D)
        return x_rot

