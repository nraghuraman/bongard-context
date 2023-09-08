from enum import Enum
import random

from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer
from .utils import (
    contrastive_loss,
    cosine_sim_loss,
    dropout_support,
    label_noise_support_pos_neg,
)


class MimicKind(Enum):
    SVM = 1
    PROTOTYPE = 2


def get_gt_rule(
    x_support_emb,
    x_query_emb,
    y_support,
    y_query,
    support_std,
    support_mean,
    kernel="linear",
    mimic_kind=MimicKind.SVM,
):
    # Normalize all embeddings
    eps = 1e-4
    x_support_emb = (x_support_emb - support_mean) / (support_std + eps)
    x_query_emb = (x_query_emb - support_mean) / (support_std + eps)

    if mimic_kind == MimicKind.SVM:
        # Compute gt
        svm_hyperplanes = []
        for x_support_b, x_query_b, y_support_b, y_query_b in zip(
            x_support_emb,
            x_query_emb,
            y_support,
            y_query,
        ):
            X = torch.cat([x_support_b, x_query_b], dim=0)
            y = torch.cat([y_support_b, y_query_b], dim=0)
            svm = SVC(kernel=kernel)
            svm.fit(X.detach().cpu().numpy(), y.detach().cpu().numpy())
            w = torch.from_numpy(svm.coef_[0]).to(device=X.device)
            b = torch.from_numpy(svm.intercept_).to(device=X.device)
            # Hyperplane (AKA rule) is the concatenation of coefficients and intercept
            svm_hyperplanes.append(torch.cat([w, b], dim=0))

        return torch.stack(svm_hyperplanes)
    elif mimic_kind == MimicKind.PROTOTYPE:
        x_all_emb = torch.cat([x_support_emb, x_query_emb], dim=1)
        y_all = torch.cat([y_support, y_query], dim=1)

        pos_prototype = torch.bmm(y_all.unsqueeze(-2).float(), x_all_emb).squeeze(1)
        pos_prototype /= torch.sum(y_all, dim=-1, keepdim=True)

        neg_prototype = torch.bmm(1 - y_all.unsqueeze(-2).float(), x_all_emb).squeeze(1)
        neg_prototype /= torch.sum(1 - y_all, dim=-1, keepdim=True)

        return pos_prototype, neg_prototype
    else:
        raise NotImplementedError(f"Mimic kind {mimic_kind} is unsupported")


class Model(nn.Module):
    """
    A support-set Transformer model, AKA "-Mimic" model, as described in the paper.
    This class implements both SVM-Mimic and Prototype-Mimic. Which one is used can be
    specified with the mimic_kind parameter.
    """

    def __init__(
        self,
        latent_dim=512,
        dropout_support_prob=0.0,
        label_noise_support_prob=0.0,
        train_encoder=False,
        temperature=1.0,
        kernel="linear",
        mimic_kind=MimicKind.SVM,
    ):
        super(Model, self).__init__()

        # Model parameters/layers
        self.cls_token_hyp = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.pos_indicator = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.neg_indicator = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        self.transformer = Transformer(
            dim=latent_dim,
            depth=6,
            heads=8,
            dim_head=64,
            mlp_dim=latent_dim,
        )

        out_dim = latent_dim + int(mimic_kind == MimicKind.SVM)
        self.rule_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_dim),
        )

        # Regularization
        self.dropout_support_prob = dropout_support_prob
        self.label_noise_support_prob = label_noise_support_prob

        # For trainable encoder
        self.temperature = temperature
        self.train_encoder = train_encoder

        # Configures SVM
        self.kernel = kernel
        assert self.kernel == "linear", "Non-linear kernels not supported for now"

        # Configures kind of mimicking (SVM or Prototype)
        self.mimic_kind = mimic_kind

    def _forward(
        self,
        x_support_emb,
        x_query_emb,
        y_support,
        y_query,
        x_support_emb_dropped,
        y_support_dropped,
        gt,
        support_std,
        support_mean,
        use_dropout,
    ):
        """
        Helper function for forward()
        """
        B, N_support, C = x_support_emb.shape

        pos_support = x_support_emb[y_support == 1].reshape(B, -1, C)
        neg_support = x_support_emb[y_support != 1].reshape(B, -1, C)
        pos_support_dropped = x_support_emb_dropped[y_support_dropped == 1].reshape(
            B, -1, C
        )
        neg_support_dropped = x_support_emb_dropped[y_support_dropped != 1].reshape(
            B, -1, C
        )

        S1 = pos_support.shape[1]
        S2 = neg_support.shape[1]
        assert S1 + S2 == N_support

        # Task 1: Train encoder
        if self.train_encoder:
            pos_query = x_query_emb[y_query == 1].reshape(B, -1, C)
            neg_query = x_query_emb[y_query != 1].reshape(B, -1, C)

            enc_loss = contrastive_loss(
                pos_support,
                pos_query,
                neg_support,
                neg_query,
                self.temperature,
            )
        else:
            enc_loss = torch.tensor(0.0, device=pos_support.device)

        # Task 2: Train rules-based Transformer
        # Task 2 First Step: Standardize all embeddings using support-set statistics
        eps = 1e-4
        pos_emb_orig = (pos_support - support_mean) / (support_std + eps)
        neg_emb_orig = (neg_support - support_mean) / (support_std + eps)
        pos_emb_orig_dropped = (pos_support_dropped - support_mean) / (
            support_std + eps
        )
        neg_emb_orig_dropped = (neg_support_dropped - support_mean) / (
            support_std + eps
        )
        x_query_emb = (x_query_emb - support_mean) / (support_std + eps)

        # Task 2 Next Step: Support dropout
        if use_dropout:
            pos_emb_input, neg_emb_input = pos_emb_orig_dropped, neg_emb_orig_dropped
        else:
            pos_emb_input, neg_emb_input = pos_emb_orig, neg_emb_orig

        # Task 2 Next Step: Label noise
        if (
            self.training
            and self.label_noise_support_prob > 0.0
            and random.random() < self.label_noise_support_prob
            and pos_emb_input.shape[1] > 2
        ):
            pos_emb_input, neg_emb_input = label_noise_support_pos_neg(
                pos_emb_input, neg_emb_input, k=1
            )

        # Task 2 Next Step: Input into the Transformer
        pos_emb = pos_emb_input.detach() + self.pos_indicator  # B,S1,C
        neg_emb = neg_emb_input.detach() + self.neg_indicator  # B,S2,C
        if self.mimic_kind == MimicKind.SVM:
            tokens = torch.cat(
                [pos_emb, neg_emb, self.cls_token_hyp.repeat(B, 1, 1)], dim=1
            )
        elif self.mimic_kind == MimicKind.PROTOTYPE:
            pos_cls = self.cls_token_hyp + self.pos_indicator
            neg_cls = self.cls_token_hyp + self.neg_indicator

            tokens = torch.cat(
                [pos_emb, neg_emb, pos_cls.repeat(B, 1, 1), neg_cls.repeat(B, 1, 1)],
                dim=1,
            )
        tokens = self.transformer(tokens)

        # Task 2 Next Step: Mimic loss to enforce student (Transformer) matches teacher
        if self.mimic_kind == MimicKind.SVM:
            rule = self.rule_head(tokens[:, -1])

            assert rule.shape == gt.shape

            mimic_loss = cosine_sim_loss(rule, gt.detach())
        elif self.mimic_kind == MimicKind.PROTOTYPE:
            pos_prototype, neg_prototype = self.rule_head(
                tokens[:, -2]
            ), self.rule_head(tokens[:, -1])
            pos_gt, neg_gt = gt

            assert pos_prototype.shape == pos_gt.shape, (
                neg_prototype.shape == neg_gt.shape
            )

            mimic_loss = cosine_sim_loss(
                pos_prototype, pos_gt.detach()
            ) + cosine_sim_loss(neg_prototype, neg_gt.detach())

        # Task 2 Next Step: Compute predictions
        if self.mimic_kind == MimicKind.SVM:
            w = rule[:, None, :-1]  # hyperplane coefficients
            b = rule[:, None, -1]  # hyperplane intercept
            query_pred_logits = torch.sum(x_query_emb * w, dim=-1) + b
            query_pred_logits /= torch.linalg.norm(w, dim=-1)
        elif self.mimic_kind == MimicKind.PROTOTYPE:
            pos_query_pred_logits = torch.bmm(
                F.normalize(x_query_emb, dim=-1),
                F.normalize(pos_prototype, dim=-1).unsqueeze(-1),
            ).squeeze(-1)
            neg_query_pred_logits = torch.bmm(
                F.normalize(x_query_emb, dim=-1),
                F.normalize(neg_prototype, dim=-1).unsqueeze(-1),
            ).squeeze(-1)
            query_pred_logits = pos_query_pred_logits - neg_query_pred_logits

        return query_pred_logits, enc_loss, mimic_loss

    def forward(self, x_support_emb, x_query_emb, y_support, y_query):
        """
        Returns:
            acc: A tensor indicating for each query in x_query_emb whether the model
                prediction is correct or not.
            logits: The predicted logits for each query.
            enc_loss: The loss on the backbone encoder. This is 0 if self.train_encoder is unset.
            mimic_loss: The loss on the support-set Transformer.
        """
        x_support_emb_dropped, y_support_dropped = dropout_support(
            x_support_emb, y_support
        )
        use_dropout = (
            self.training
            and self.dropout_support_prob > 0
            and random.random() < self.dropout_support_prob
        )
        if use_dropout:
            support_std, support_mean = torch.std_mean(
                x_support_emb_dropped, dim=-2, keepdims=True
            )
        else:
            support_std, support_mean = torch.std_mean(
                x_support_emb, dim=-2, keepdims=True
            )

        gt = get_gt_rule(
            x_support_emb,
            x_query_emb,
            y_support,
            y_query,
            support_std,
            support_mean,
            kernel=self.kernel,
            mimic_kind=self.mimic_kind,
        )
        logits, enc_loss, mimic_loss = self._forward(
            x_support_emb,
            x_query_emb,
            y_support,
            y_query,
            x_support_emb_dropped,
            y_support_dropped,
            gt,
            support_std,
            support_mean,
            use_dropout,
        )

        acc = ((logits >= 0) == y_query).float()
        return acc, logits, enc_loss, mimic_loss


def get_model(args):
    return Model(
        latent_dim=args.latent_dim,
        dropout_support_prob=args.dropout_support_prob,
        label_noise_support_prob=args.label_noise_support_prob,
        temperature=args.temperature,
        train_encoder=args.train_encoder,
        kernel=args.kernel,
        mimic_kind=getattr(MimicKind, args.mimic_kind),
    ).to(device=args.device)
