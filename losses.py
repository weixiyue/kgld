"""Flight-summed negative ELBO and the approved ordinal HI objective."""

import torch
from torch.nn import functional as F


def diagonal_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * (logvar_p - logvar_q
                  + torch.exp(logvar_q - logvar_p)
                  + (mu_q - mu_p).square() * torch.exp(-logvar_p) - 1).sum(dim=-1)


def reconstruction_loss(x, x_hat, mask, sigma_x_squared):
    if sigma_x_squared <= 0:
        raise ValueError("sigma_x_squared must be positive")
    if x.shape != x_hat.shape or x.shape[:2] != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Reconstruction tensors and bool mask have incompatible shapes")
    # Select valid elements before arithmetic, so padding cannot contaminate loss.
    residual = x.masked_fill(~mask[:, :, None], 0) - x_hat.masked_fill(~mask[:, :, None], 0)
    return residual.square().sum(dim=(1, 2)) / (2 * sigma_x_squared)


def representation_loss(x, mask, output, sigma_x_squared):
    reconstruction = reconstruction_loss(x, output["x_hat"], mask, sigma_x_squared)
    kl_i = diagonal_gaussian_kl(output["mu_i"], output["logvar_i"],
                               torch.zeros_like(output["mu_i"]), torch.zeros_like(output["logvar_i"]))
    kl_c = diagonal_gaussian_kl(output["mu_c"], output["logvar_c"],
                               output["prior_mu_c"], output["prior_logvar_c"])
    return {"loss": (reconstruction + kl_i + kl_c).mean(),
            "reconstruction": reconstruction.mean(), "kl_i": kl_i.mean(), "kl_c": kl_c.mean()}


def order_loss(r_early, r_late, kappa):
    if kappa <= 0 or r_early.ndim != 1 or r_early.shape != r_late.shape or r_early.numel() == 0:
        raise ValueError("Ordering requires nonempty matching score vectors and positive kappa")
    return F.softplus((r_early - r_late) / kappa).mean()


def head_l2(hi_head):
    return torch.stack([parameter.square().sum() for parameter in hi_head.parameters()]).sum()


def hi_loss(r_early, r_late, hi_head, kappa, l2_coefficient):
    if l2_coefficient < 0:
        raise ValueError("l2_coefficient must be nonnegative")
    order = order_loss(r_early, r_late, kappa)
    l2 = head_l2(hi_head)
    return {"loss": order + l2_coefficient * l2, "order": order, "l2": l2}
