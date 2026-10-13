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


def representation_loss(x, mask, output, sigma_x_squared, weights=None):
    reconstruction = reconstruction_loss(x, output["x_hat"], mask, sigma_x_squared)
    kl_i = diagonal_gaussian_kl(output["mu_i"], output["logvar_i"],
                               torch.zeros_like(output["mu_i"]), torch.zeros_like(output["logvar_i"]))
    kl_c = diagonal_gaussian_kl(output["mu_c"], output["logvar_c"],
                               output["prior_mu_c"], output["prior_logvar_c"])
    terms = {"reconstruction": reconstruction.mean(), "kl_i": kl_i.mean(), "kl_c": kl_c.mean()}
    if weights is None or all(weights[key] == 1 for key in terms):
        loss = (reconstruction + kl_i + kl_c).mean()
    else:
        active = [weights[key] * value for key, value in terms.items() if weights[key] != 0]
        loss = sum(active[1:], active[0]) if active else reconstruction.new_zeros(())
    return {"loss": loss, **terms}


def order_loss(r_early, r_late, kappa):
    if kappa <= 0 or r_early.ndim != 1 or r_early.shape != r_late.shape or r_early.numel() == 0:
        raise ValueError("Ordering requires nonempty matching score vectors and positive kappa")
    return F.softplus((r_early - r_late) / kappa).mean()


def head_l2(hi_head):
    return torch.stack([parameter.square().sum() for parameter in hi_head.parameters()]).sum()


def hi_loss(r_early, r_late, hi_head, kappa, l2_coefficient):
    if l2_coefficient < 0:
        raise ValueError("l2_coefficient must be nonnegative")
    weights = {'reconstruction': 0., 'kl_i': 0., 'kl_c': 0., 'order': 1., 'l2': l2_coefficient}
    terms = training_loss(hi_head, weights, r_early=r_early, r_late=r_late, kappa=kappa)
    return {key: terms[key] for key in ('loss', 'order', 'l2')}


def training_loss(head, weights, *, x=None, mask=None, output=None,
                  r_early=None, r_late=None, sigma_x_squared=1., kappa=1.):
    """One objective for the three flows; log raw terms, exclude zero-weight graphs.

    Unavailable terms (e.g. reconstruction in frozen Stage 2) are zero. Stage 1
    retains its original per-flight sum before averaging, including rounding.
    """
    keys = ('reconstruction', 'kl_i', 'kl_c', 'order', 'l2')
    if set(weights) != set(keys) or any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in weights.values()):
        raise ValueError('Loss weights must specify five finite nonnegative coefficients')
    if head is None and weights["l2"] != 0:
        raise ValueError("No HIHead: l2 weight must be zero")
    reference = output["mu_i"] if output is not None else r_early
    zero = next(head.parameters()).new_zeros(()) if head is not None else reference.new_zeros(())
    terms = dict.fromkeys(keys, zero)
    active = []
    if output is not None:
        representation = representation_loss(x, mask, output, sigma_x_squared, weights)
        terms.update({key: representation[key] for key in keys[:3]})
        if any(weights[key] for key in keys[:3]):
            active.append(representation['loss'])
    elif any(weights[key] for key in keys[:3]):
        raise ValueError('Representation inputs are required for nonzero representation weights')
    if r_early is not None and r_early.numel():
        terms['order'] = order_loss(r_early, r_late, kappa)
    elif weights['order']:
        raise ValueError('A nonzero order weight requires valid pairs')
    terms['l2'] = head_l2(head) if head is not None else zero
    for key in ('order', 'l2'):
        if weights[key]:
            active.append(weights[key] * terms[key])
    loss = sum(active[1:], active[0]) if active else zero
    return {'loss': loss, **terms}
