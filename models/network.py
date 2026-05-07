import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ContinuousTimeEncoding, StepwiseModalityFusion, TemporalAttention
from .shufflenetv2 import shufflenet_v2_x0_5


class SelectiveGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(channels // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class TemporalAlignedNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.get("model", {})
        self.d = cfg["model_dims"]["latent_space_d"]
        self.normalize_modalities = model_cfg.get("normalize_modalities", True)

        dropout = model_cfg.get("dropout", 0.4)
        time_dropout = model_cfg.get("time_dropout", 0.5)
        head_dropout = model_cfg.get("head_dropout", 0.5)

        self.m1_roi_enc = shufflenet_v2_x0_5()
        self.m1_ctx_enc = shufflenet_v2_x0_5()
        self.roi_selector = SelectiveGate(1024)
        self.ctx_selector = SelectiveGate(1024)

        self.m1_proj = nn.Sequential(
            nn.Linear(2048, self.d),
            nn.LayerNorm(self.d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.m2_enc = nn.Sequential(
            nn.Linear(cfg["model_dims"]["m2_input_dim"], self.d),
            nn.LayerNorm(self.d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.m3_enc = nn.Sequential(
            nn.Linear(cfg["model_dims"]["m3_input_dim"], self.d),
            nn.LayerNorm(self.d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.step_fusion = StepwiseModalityFusion(
            self.d,
            num_modalities=3,
            temperature=model_cfg.get("fusion_temperature", 2.0),
            prior_bias=model_cfg.get("fusion_prior_bias", [0.5, -0.5, 0.2]),
        )
        self.time_encoding = ContinuousTimeEncoding(self.d)
        self.time_alpha = nn.Parameter(torch.zeros(1))
        self.time_dropout = nn.Dropout(time_dropout)
        self.global_temp_attn = TemporalAttention(self.d)

        self.head_dropout = nn.Dropout(head_dropout)
        self.mpr_head = nn.Linear(self.d, 1)
        self.cox_head = nn.Linear(self.d, 1)

    def forward(self, m1_roi, m1_ctx, m2, m3, global_mask, m1_avail, m2_avail, m3_avail, time_deltas):
        batch_size, time_steps = m1_roi.size(0), m1_roi.size(1)

        roi_in = m1_roi.view(batch_size * time_steps, *m1_roi.shape[2:])
        ctx_in = m1_ctx.view(batch_size * time_steps, *m1_ctx.shape[2:])
        f_roi = self.roi_selector(self.m1_roi_enc(roi_in))
        f_ctx = self.ctx_selector(self.m1_ctx_enc(ctx_in))

        h1 = self.m1_proj(torch.cat([f_roi, f_ctx], dim=1)).view(batch_size, time_steps, self.d)
        h2 = self.m2_enc(m2)
        h3 = self.m3_enc(m3)

        if self.normalize_modalities:
            h1 = F.normalize(h1, p=2, dim=-1)
            h2 = F.normalize(h2, p=2, dim=-1)
            h3 = F.normalize(h3, p=2, dim=-1)

        h1 = h1 * m1_avail.unsqueeze(-1).float()
        h2 = h2 * m2_avail.unsqueeze(-1).float()
        h3 = h3 * m3_avail.unsqueeze(-1).float()

        modality_stack = torch.stack([h1, h2, h3], dim=2)
        availability_stack = torch.stack([m1_avail, m2_avail, m3_avail], dim=2)
        fused_sequence, _ = self.step_fusion(modality_stack, availability_stack)

        time_encoding = self.time_encoding(time_deltas)
        time_aware_sequence = fused_sequence + self.time_alpha * self.time_dropout(time_encoding)
        patient_embedding, _ = self.global_temp_attn(time_aware_sequence, global_mask)

        patient_embedding = self.head_dropout(patient_embedding)
        return self.mpr_head(patient_embedding), self.cox_head(patient_embedding)
