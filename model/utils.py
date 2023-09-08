import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(pos_support, pos_query, neg_support, neg_query, temperature):
    """
    Contrastive loss for training the encoder, described in equation 1 in the paper.
    """
    B, S1, _ = pos_support.shape
    B, S2, _ = neg_support.shape
    B, Q1, _ = pos_query.shape
    B, Q2, _ = neg_query.shape

    feats = torch.cat([pos_support, pos_query, neg_support, neg_query], dim=1)
    norm_feats = F.normalize(feats, dim=-1)

    sims = torch.bmm(norm_feats, torch.transpose(norm_feats, 1, 2)) / temperature
    assert sims.shape == (B, S1 + S2 + Q1 + Q2, S1 + S2 + Q1 + Q2)

    n_pos = S1 + Q1
    pos_to_neg = torch.sum(torch.exp(sims[:, :n_pos, n_pos:]), dim=-1, keepdim=True)
    pos_to_pos = torch.exp(sims[:, :n_pos, :n_pos])
    # Remove the diagonal elements,
    # from https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    pos_to_pos = (
        pos_to_pos.flatten(start_dim=1)[:, 1:]
        .view(B, n_pos - 1, n_pos + 1)[:, :, :-1]
        .reshape(B, n_pos, n_pos - 1)
    )
    probs = pos_to_pos / (pos_to_pos + pos_to_neg)

    return -torch.mean(torch.log(probs))


def protonet_loss(x_support_emb, x_query_emb, y_support_emb, y_query_emb, scale, bias):
    """
    ProtoNet loss from "Prototypical Networks for Few-Shot Learning" by Snell et al., as used
    by Hu et al. in "Pushing the Limits of Simple Pipelines for Few-Shot Learning."

    Code adapted from https://github.com/hushell/pmf_cvpr22/blob/main/models/protonet.py
    """

    def cos_classifier(w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2)  # B, M, nC
        cls_scores = scale * (cls_scores + bias)
        return cls_scores

    num_classes = torch.max(y_support_emb).long() + 1

    y_support_1hot = F.one_hot(y_support_emb, num_classes).transpose(
        1, 2
    )  # B, nC, nSupp

    # B, nC, nSupp x B, nSupp, d = B, nC, d
    prototypes = torch.bmm(y_support_1hot.float(), x_support_emb)
    prototypes = prototypes / y_support_1hot.sum(
        dim=2, keepdim=True
    )  # NOTE: may div 0 if some classes got 0 images

    logits = cos_classifier(prototypes, x_query_emb)  # B, nQry, nC

    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        logits.view(logits.shape[0] * logits.shape[1], -1), y_query_emb.view(-1)
    )

    return (
        loss,
        logits,
    )


def cosine_sim_loss(input_, target):
    """
    Cosine similarity loss for training a support-set Transformer, described in
    equations 2 and 3 in the paper.
    """
    return 1 - torch.mean(
        torch.sum(F.normalize(target, dim=-1) * F.normalize(input_, dim=-1), dim=-1)
    )


def dropout_support(x_support, y_support, rng=None, k=None):
    """
    Implements the "support dropout" procedure described in the "Method" section
    of the paper.
    """
    B = x_support.shape[0]
    N_classes = torch.max(y_support).int() + 1
    x_dropped = []
    y_dropped = []

    prev = None

    # Written to generalize to multiclass case.
    # Assumes that all classes have same number of supports.
    for i in range(N_classes):
        class_support = x_support[y_support == i].reshape(B, -1, *x_support.shape[2:])
        S = class_support.shape[1]

        if prev is not None:  # Not the first class that we're looking at
            num_class = prev
            assert num_class <= S
        else:  # The first class that we're looking at
            if k != None:
                num_class = min(k, S)
            elif S <= 2:
                num_class = S
            elif rng is not None:
                num_class = rng.randint(2, S + 1)
            else:
                num_class = np.random.randint(2, S + 1)
        prev = num_class

        class_indices = torch.arange(S)

        if rng is not None:
            perm = rng.permutation(class_indices.shape[0])
        else:
            perm = np.random.permutation(class_indices.shape[0])

        class_indices = class_indices[perm[:num_class]]

        x_dropped.append(class_support[:, class_indices])
        y_dropped.append(torch.ones((B, num_class), device=x_support.device) * i)

    return torch.cat(x_dropped, dim=1), torch.cat(y_dropped, dim=1)


def label_noise_support_pos_neg(pos_sup, neg_sup, rng=None, k=1):
    """
    Implements the "label noise" procedure described in the "Method" section of the paper.
    Unlike label_noise_support(), directly accepts positive and negative supports.
    """
    if rng is not None:
        pos_noise = rng.choice(pos_sup.shape[1], size=k, replace=False)
        neg_noise = rng.choice(neg_sup.shape[1], size=k, replace=False)
    else:
        pos_noise = np.random.choice(pos_sup.shape[1], size=k, replace=False)
        neg_noise = np.random.choice(neg_sup.shape[1], size=k, replace=False)

    pos_sup[:, pos_noise], neg_sup[:, neg_noise] = (
        neg_sup[:, neg_noise],
        pos_sup[:, pos_noise],
    )

    return pos_sup, neg_sup


def label_noise_support(x_support, y_support, rng=None, k=1):
    """
    Implements the "label noise" procedure described in the "Method" section of the paper.
    """
    B = x_support.shape[0]

    pos_sup = x_support[y_support == 1].reshape(B, -1, *x_support.shape[2:])
    neg_sup = x_support[y_support == 0].reshape(B, -1, *x_support.shape[2:])

    pos_sup, neg_sup = label_noise_support_pos_neg(pos_sup, neg_sup, rng=rng, k=k)

    x_support = torch.cat([pos_sup, neg_sup], dim=1)
    y_support = torch.cat(
        [torch.ones(pos_sup.shape[:2]), torch.zeros(neg_sup.shape[:2])], dim=1
    ).to(dtype=torch.long, device=x_support.device)

    return x_support, y_support


# Copied from https://github.com/hushell/pmf_cvpr22/blob/main/models/utils.py
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
