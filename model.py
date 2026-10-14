"""Approved OCALD GRU representation and deterministic HI head."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


HI_MAPPING_TYPE = "dev_score_reverse_minmax"


def fit_score_scale(dev_scores):
    """Fit one global mapping from all final dev scores, without labels."""
    if dev_scores.ndim != 1 or not len(dev_scores) or not torch.isfinite(dev_scores).all():
        raise ValueError("HI mapping requires nonempty finite dev scores")
    minimum, maximum = dev_scores.min().item(), dev_scores.max().item()
    return {"type": HI_MAPPING_TYPE, "min": minimum, "max": maximum,
            "fit_split": "dev", "count": len(dev_scores), "available": maximum != minimum}


def validate_score_scale(parameters):
    if (parameters.get("type") != HI_MAPPING_TYPE or parameters.get("fit_split") != "dev"
            or not isinstance(parameters.get("count"), int) or parameters["count"] <= 0
            or not all(torch.isfinite(torch.tensor(parameters[key], dtype=torch.float64)) for key in ("min", "max"))
            or parameters["max"] < parameters["min"]
            or parameters.get("available") != (parameters["max"] != parameters["min"])):
        raise ValueError("Invalid dev score HI mapping; legacy HI Min-Max is not accepted")


def scale_score(scores, parameters):
    """The single HI mapping used by training, inference and evaluation."""
    validate_score_scale(parameters)
    if not parameters["available"]:
        raise ValueError("Dev score range is zero; HI is unavailable")
    return (parameters["max"] - scores) / (parameters["max"] - parameters["min"])


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


def sequence_states(gru, sequence, mask):
    """Packed GRU states; unpack to the supplied padding length."""
    lengths = valid_lengths(mask)
    if sequence.ndim != 3 or sequence.shape[:2] != mask.shape or sequence.device != mask.device:
        raise ValueError("Sequence and mask must share batch/time dimensions and device")
    packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, hidden = gru(packed)
    states, _ = pad_packed_sequence(output, batch_first=True, total_length=mask.shape[1])
    return hidden[-1], states


class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_i_dim, latent_c_dim, c_mode="global", condition_hidden_dim=32):
        super().__init__()
        self.c_mode = c_mode
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu_i = nn.Linear(hidden_dim, latent_i_dim)
        self.logvar_i = nn.Linear(hidden_dim, latent_i_dim)
        c_input_dim = hidden_dim + condition_hidden_dim if c_mode == "temporal" else hidden_dim
        self.mu_c = nn.Linear(c_input_dim, latent_c_dim)
        self.logvar_c = nn.Linear(c_input_dim, latent_c_dim)

    def forward(self, x, mask):
        state = last_valid_state(self.gru, x, mask)
        if self.c_mode == "temporal":
            return self.mu_i(state), self.logvar_i(state)
        return self.mu_i(state), self.logvar_i(state), self.mu_c(state), self.logvar_c(state)

    def temporal_states(self, x, mask):
        last, states = sequence_states(self.gru, x, mask)
        return self.mu_i(last), self.logvar_i(last), states

    def temporal_posterior(self, h, u, mask):
        features = torch.cat((h, u), dim=-1)
        return tuple(head(features).masked_fill(~mask[:, :, None], 0) for head in (self.mu_c, self.logvar_c))


class ConditionalPrior(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_c_dim, c_mode="global"):
        super().__init__()
        self.c_mode = c_mode
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_c_dim)
        self.logvar = nn.Linear(hidden_dim, latent_c_dim)

    def temporal_states(self, w, mask):
        _, states = sequence_states(self.gru, w, mask)
        mu, logvar = (head(states).masked_fill(~mask[:, :, None], 0) for head in (self.mu, self.logvar))
        return states, mu, logvar

    def forward(self, w, mask):
        if self.c_mode == "temporal":
            _, mu, logvar = self.temporal_states(w, mask)
            return mu, logvar
        state = last_valid_state(self.gru, w, mask)
        return self.mu(state), self.logvar(state)


class Decoder(nn.Module):
    def __init__(self, latent_i_dim, latent_c_dim, hidden_dims, output_dim):
        super().__init__()
        self.use_w = False
        self.condition_dim = 0
        self.layers = nn.Sequential(
            nn.Linear(latent_i_dim + latent_c_dim + 1, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def enable_w_input(self, condition_dim):
        # Initialize only the resized layer, without advancing the common RNG.
        with torch.random.fork_rng(devices=[]):
            old = self.layers[0]
            self.layers[0] = nn.Linear(old.in_features + condition_dim, old.out_features)
        self.use_w = True
        self.condition_dim = condition_dim

    def forward(self, z_i, z_c, positions, w=None):
        if z_c.ndim == 3:
            if z_c.shape[:2] != positions.shape:
                raise ValueError("Temporal C must align with per-step positions")
            latents = torch.cat((z_i[:, None, :].expand(-1, positions.shape[1], -1), z_c), dim=-1)
            parts = [latents]
        else:
            latents = torch.cat((z_i, z_c), dim=-1)
            parts = [latents[:, None, :].expand(-1, positions.shape[1], -1)]
        if self.use_w:
            if w is None or w.shape != (*positions.shape, self.condition_dim):
                raise ValueError('W-conditioned decoder requires aligned per-step W')
            parts.append(w)
        parts.append(positions[:, :, None])
        inputs = torch.cat(parts, dim=-1)
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
        self.c_mode = m.get("c_mode", "global")
        if self.c_mode not in ("global", "temporal"):
            raise ValueError("Unknown c_mode")
        approved = {
            "sensor_hidden_dim": 64, "condition_hidden_dim": 32, "gru_layers": 1,
            "bidirectional": False, "latent_i_dim": m["latent_i_dim"], "latent_c_dim": 2 if self.c_mode == "temporal" else 8,
            "decoder_hidden_dims": [64, 64], "hi_hidden_dim": 16,
        }
        self.score_mode = m.get('score_mode', 'hi_head')
        if self.score_mode not in ('hi_head', 'scalar_i'):
            raise ValueError('Unknown score_mode')
        if m['latent_i_dim'] != (1 if self.score_mode == 'scalar_i' else 8):
            raise ValueError('score_mode and latent_i_dim mismatch')
        self.decoder_use_w = m.get('decoder_use_w', False)
        if not isinstance(self.decoder_use_w, bool):
            raise ValueError('decoder_use_w must be boolean')
        if {k: v for k, v in m.items() if k not in ('decoder_use_w', 'score_mode', 'c_mode')} != approved:
            raise ValueError("Model settings differ from approved v2 structure")
        if self.c_mode == "temporal" and (self.decoder_use_w or self.score_mode != "hi_head"):
            raise ValueError("Temporal C requires 8D I/HIHead and no decoder W input")
        self.sensor_encoder = SensorEncoder(d["sensor_dim"], m["sensor_hidden_dim"], m["latent_i_dim"], m["latent_c_dim"], self.c_mode, m["condition_hidden_dim"])
        self.conditional_prior = ConditionalPrior(d["condition_dim"], m["condition_hidden_dim"], m["latent_c_dim"], self.c_mode)
        self.decoder = Decoder(m["latent_i_dim"], m["latent_c_dim"], m["decoder_hidden_dims"], d["sensor_dim"])
        self.hi_head = HIHead(m["latent_i_dim"], m["hi_hidden_dim"]) if self.score_mode == "hi_head" else None
        if self.decoder_use_w:
            self.decoder.enable_w_input(d['condition_dim'])

    def encode(self, x, mask):
        return self.sensor_encoder(x, mask)

    def decode(self, z_i, z_c, positions, w=None):
        return self.decoder(z_i, z_c, positions, w)

    def score(self, mu_i):
        return mu_i.squeeze(-1) if self.score_mode == "scalar_i" else self.hi_head(mu_i)

    def predict_score(self, x, mask):
        return self.score(self.encode(x, mask)[0])

    def predict_hi(self, x, mask, score_parameters):
        return scale_score(self.predict_score(x, mask), score_parameters)

    def posterior(self, x, w, mask):
        if self.c_mode == "temporal":
            mu_i, logvar_i, h = self.sensor_encoder.temporal_states(x, mask)
            u, prior_mu_c, prior_logvar_c = self.conditional_prior.temporal_states(w, mask)
            mu_c, logvar_c = self.sensor_encoder.temporal_posterior(h, u, mask)
        else:
            mu_i, logvar_i, mu_c, logvar_c = self.encode(x, mask)
            prior_mu_c, prior_logvar_c = self.conditional_prior(w, mask)
        return {"mu_i": mu_i, "logvar_i": logvar_i, "mu_c": mu_c, "logvar_c": logvar_c,
                "prior_mu_c": prior_mu_c, "prior_logvar_c": prior_logvar_c}

    def forward(self, x, w, mask):
        output = self.posterior(x, w, mask)
        z_i = output["mu_i"] + torch.exp(0.5 * output["logvar_i"]) * torch.randn_like(output["mu_i"])
        z_c = output["mu_c"] + torch.exp(0.5 * output["logvar_c"]) * torch.randn_like(output["mu_c"])
        decoder_w = w.masked_fill(~mask[:, :, None], 0) if self.decoder_use_w else None
        output["x_hat"] = self.decode(z_i, z_c, time_positions(mask, dtype=x.dtype), decoder_w)
        return output
