"""Approved OCALD GRU representation and deterministic HI head."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def valid_lengths(mask):
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B, T]")
    lengths = mask.sum(dim=1)
    if len(lengths) == 0 or (lengths == 0).any():
        raise ValueError("Each flight must have at least one valid time step")
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("mask must describe a valid prefix followed by padding")
    return lengths


def time_positions(mask, dtype=torch.float32):
    lengths = valid_lengths(mask)
    positions = torch.arange(mask.shape[1], device=mask.device, dtype=dtype)[None, :]
    # A one-step flight has time position zero. Padding is set to zero.
    return (positions / (lengths - 1).clamp_min(1)[:, None]).masked_fill(~mask, 0)


def last_valid_state(gru, sequence, mask):
    lengths = valid_lengths(mask)
    if sequence.ndim != 3 or sequence.shape[:2] != mask.shape or sequence.device != mask.device:
        raise ValueError("Sequence and mask must share batch/time dimensions and device")
    packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, hidden = gru(packed)
    return hidden[-1]


class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_i_dim, latent_c_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu_i = nn.Linear(hidden_dim, latent_i_dim)
        self.logvar_i = nn.Linear(hidden_dim, latent_i_dim)
        self.mu_c = nn.Linear(hidden_dim, latent_c_dim)
        self.logvar_c = nn.Linear(hidden_dim, latent_c_dim)

    def forward(self, x, mask):
        state = last_valid_state(self.gru, x, mask)
        return self.mu_i(state), self.logvar_i(state), self.mu_c(state), self.logvar_c(state)


class ConditionalPrior(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_c_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_c_dim)
        self.logvar = nn.Linear(hidden_dim, latent_c_dim)

    def forward(self, w, mask):
        state = last_valid_state(self.gru, w, mask)
        return self.mu(state), self.logvar(state)


class Decoder(nn.Module):
    def __init__(self, latent_i_dim, latent_c_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_i_dim + latent_c_dim + 1, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, z_i, z_c, positions):
        latents = torch.cat((z_i, z_c), dim=-1)
        inputs = torch.cat((latents[:, None, :].expand(-1, positions.shape[1], -1),
                            positions[:, :, None]), dim=-1)
        return self.layers(inputs)


class HIHead(nn.Module):
    def __init__(self, latent_i_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_i_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, mu_i):
        return self.layers(mu_i).squeeze(-1)


class OCALD(nn.Module):
    def __init__(self, config):
        super().__init__()
        m, d = config["model"], config["data"]
        approved = {
            "sensor_hidden_dim": 64, "condition_hidden_dim": 32, "gru_layers": 1,
            "bidirectional": False, "latent_i_dim": 8, "latent_c_dim": 8,
            "decoder_hidden_dims": [64, 64], "hi_hidden_dim": 16,
        }
        if m != approved:
            raise ValueError("Model settings differ from approved v2 structure")
        self.sensor_encoder = SensorEncoder(d["sensor_dim"], m["sensor_hidden_dim"], m["latent_i_dim"], m["latent_c_dim"])
        self.conditional_prior = ConditionalPrior(d["condition_dim"], m["condition_hidden_dim"], m["latent_c_dim"])
        self.decoder = Decoder(m["latent_i_dim"], m["latent_c_dim"], m["decoder_hidden_dims"], d["sensor_dim"])
        self.hi_head = HIHead(m["latent_i_dim"], m["hi_hidden_dim"])

    def encode(self, x, mask):
        return self.sensor_encoder(x, mask)

    def decode(self, z_i, z_c, positions):
        return self.decoder(z_i, z_c, positions)

    def predict_hi(self, x, mask):
        mu_i, _, _, _ = self.encode(x, mask)
        return torch.sigmoid(-self.hi_head(mu_i))

    def forward(self, x, w, mask):
        mu_i, logvar_i, mu_c, logvar_c = self.encode(x, mask)
        prior_mu_c, prior_logvar_c = self.conditional_prior(w, mask)
        z_i = mu_i + torch.exp(0.5 * logvar_i) * torch.randn_like(mu_i)
        z_c = mu_c + torch.exp(0.5 * logvar_c) * torch.randn_like(mu_c)
        reconstruction = self.decode(z_i, z_c, time_positions(mask, dtype=x.dtype))
        return {"mu_i": mu_i, "logvar_i": logvar_i, "mu_c": mu_c, "logvar_c": logvar_c,
                "prior_mu_c": prior_mu_c, "prior_logvar_c": prior_logvar_c, "x_hat": reconstruction}
