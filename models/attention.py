import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(1, hidden_dim))
        self.phi = nn.Parameter(torch.randn(1, hidden_dim))
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, time_deltas):
        t = time_deltas.unsqueeze(-1)
        time_encoding = torch.sin(t * self.omega + self.phi)
        return self.linear(time_encoding)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, mask):
        scores = self.attention_net(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        alpha = F.softmax(scores, dim=-1)
        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return pooled, alpha


class StepwiseModalityFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=3, temperature=2.0, prior_bias=None):
        super().__init__()
        if prior_bias is None:
            prior_bias = [0.5, -0.5, 0.2]
        if len(prior_bias) != num_modalities:
            raise ValueError("prior_bias length must match num_modalities")

        self.temperature = temperature
        self.scoring_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.prior_bias = nn.Parameter(torch.tensor(prior_bias, dtype=torch.float32))

    def forward(self, modality_stack, availability_stack):
        batch_size, time_steps, num_modalities, hidden_dim = modality_stack.size()
        flat_modalities = modality_stack.view(batch_size * time_steps, num_modalities, hidden_dim)
        flat_availability = availability_stack.view(batch_size * time_steps, num_modalities)

        scores = self.scoring_fn(flat_modalities).squeeze(-1)
        scores = (scores + self.prior_bias) / self.temperature
        scores = scores.masked_fill(~flat_availability, -1e4)

        beta = F.softmax(scores, dim=-1)
        fused_step = torch.bmm(beta.unsqueeze(1), flat_modalities).squeeze(1)
        return fused_step.view(batch_size, time_steps, hidden_dim), beta.view(batch_size, time_steps, num_modalities)
