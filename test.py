from collections import defaultdict
import os.path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_loaders
from model.encoder import get_encoder, encode
from model.utils import dropout_support, label_noise_support
from utils.args import get_args
from utils.baselines import Baseline
from utils.tools import log


# Some code adapted from PMF codebase
def cos_classifier(w, f):
    """
    w.shape = B, nC, d
    f.shape = B, M, d
    """
    f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
    w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

    cls_scores = f @ w.transpose(1, 2)  # B, M, nC
    return cls_scores


def eval_(
    model,
    encoder,
    args,
    test_dataloader,
    dataset_name,
    eval_dropout=-1,
    eval_label_noise=-1,
    baseline=Baseline.NONE,
):
    model.eval()
    encoder.eval()

    total = 0
    correct = 0
    total_enc_loss = 0
    total_mimic_loss = 0
    num_iters = 0

    if eval_dropout != -1 or eval_label_noise != -1:
        rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            (
                x_support,
                x_query,
                y_support,
                y_query,
            ) = batch
            x_support = x_support.to(device=args.device)
            x_query = x_query.to(device=args.device)
            y_support = y_support.to(device=args.device)
            y_query = y_query.to(device=args.device)

            if eval_dropout != -1:
                x_support, y_support = dropout_support(
                    x_support, y_support, rng=rng, k=eval_dropout
                )
            if eval_label_noise != -1:
                x_support, y_support = label_noise_support(
                    x_support, y_support, rng=rng, k=eval_label_noise
                )

            x_support_emb, x_query_emb = encode(
                encoder, x_support, x_query, use_grad=False
            )

            if baseline != Baseline.NONE:
                # Normalization. Note that if baseline == Baseline.NONE, the model
                # should already perform any normalization if needed.
                if args.eval_standardize:
                    support_std, support_mean = torch.std_mean(
                        x_support_emb, dim=-2, keepdims=True
                    )
                    eps = 1e-4

                    x_support_emb = (x_support_emb - support_mean) / (support_std + eps)
                    x_query_emb = (x_query_emb - support_mean) / (support_std + eps)

                # For replicating ablations with different forms of normalization
                elif args.eval_standardize_train:
                    eps = 1e-4

                    x_support_emb = (x_support_emb - args.mean) / (args.std + eps)
                    x_query_emb = (x_query_emb - args.mean) / (args.std + eps)
                elif args.eval_normalize_l2:
                    x_support_emb = F.normalize(x_support_emb, dim=-1)
                    x_query_emb = F.normalize(x_query_emb, dim=-1)

            if baseline == Baseline.NONE:
                acc, _, enc_loss, mimic_loss = model(
                    x_support_emb, x_query_emb, y_support, y_query
                )
                total_enc_loss += enc_loss.item()
                total_mimic_loss += mimic_loss.item()
            elif baseline == Baseline.SVM:
                assert (
                    args.batch_size == 1
                ), "For simplicity, batch size of 1 is required for SVM baseline"

                # One-vs-rest classification
                # Written to generalize to multiclass case (unneeded for Bongard problems)
                num_classes = torch.max(y_support) + 1
                all_ests = []
                for i in range(num_classes.long().item()):
                    if num_classes == 2 and i == 0:
                        continue  # Only need 1 hyperplane in this case

                    pos_sup = x_support_emb[y_support == i].reshape(
                        x_support_emb.shape[0], -1, *x_support_emb.shape[2:]
                    )
                    neg_sup = x_support_emb[y_support != i].reshape(
                        x_support_emb.shape[0], -1, *x_support_emb.shape[2:]
                    )

                    all_supports = torch.cat([pos_sup, neg_sup], dim=1)

                    X = all_supports.squeeze(0)
                    y = torch.cat(
                        [torch.ones(pos_sup.shape[1]), torch.zeros(neg_sup.shape[1])],
                        dim=0,
                    )

                    assert (
                        args.kernel == "linear"
                    ), "Non-linear kernels not yet supported"
                    svm = SVC(kernel=args.kernel)
                    svm.fit(X.detach().cpu().numpy(), y.detach().cpu().numpy())
                    w = torch.from_numpy(svm.coef_[0]).to(
                        device=X.device
                    )  # Coefficients
                    b = torch.from_numpy(svm.intercept_).to(
                        device=X.device
                    )  # Intercept
                    est = (
                        torch.sum(x_query_emb.squeeze(0) * w, dim=-1) + b
                    ) / torch.linalg.norm(w, dim=-1)
                    all_ests.append(est)

                if num_classes > 2:
                    all_ests = torch.stack(all_ests, dim=0)
                    acc = (torch.argmax(all_ests, dim=0) == y_query).float()
                else:
                    acc = ((all_ests[0] >= 0) == y_query).float()
            elif baseline == Baseline.PROTOTYPE:
                num_classes = torch.max(y_support).long().item() + 1

                y_support_1hot = F.one_hot(y_support.long(), num_classes).transpose(
                    1, 2
                )  # B, nC, nSupp

                # B, nC, nSupp x B, nSupp, d = B, nC, d
                prototypes = torch.bmm(y_support_1hot.float(), x_support_emb)
                prototypes = prototypes / y_support_1hot.sum(
                    dim=2, keepdim=True
                )  # NOTE: may div 0 if some classes got 0 images

                all_ests = cos_classifier(prototypes, x_query_emb)  # B, nQry, nC

                acc = (torch.argmax(all_ests, dim=2) == y_query).float()
            elif baseline == Baseline.KNN:
                B, nQry, _ = x_query_emb.shape

                num_classes = torch.max(y_support).long().item() + 1
                assert (
                    num_classes == 2
                ), "knn currently only implemented for binary classification"

                similarities = cos_classifier(
                    x_support_emb, x_query_emb
                )  # B, nQry, nSup
                indices = torch.argsort(-1 * similarities, dim=-1)
                closest_support_labels = (
                    torch.gather(
                        y_support, dim=-1, index=indices[:, :, : args.k].reshape(B, -1)
                    )
                    .reshape(B, nQry, -1)
                    .float()
                )
                all_ests = (torch.mean(closest_support_labels, dim=-1) > 0.5).long()

                acc = (all_ests == y_query).float()
            else:
                raise NotImplementedError("Other baselines not yet implemented")

            total += 1
            correct += torch.mean(acc).item()

            num_iters += 1

    accuracy = correct / total

    log.info(f"Accuracy on val set: {accuracy}")
    log.info(f"Enc loss on val set: {total_enc_loss / num_iters}")
    log.info(f"mimic loss on val set: {total_mimic_loss / num_iters}")

    if args.writer:
        args.writer.add_scalar(
            f"Enc_loss/val_{dataset_name}", total_enc_loss / num_iters, args.curr_iter
        )
        args.writer.add_scalar(
            f"Mimic_loss/val_{dataset_name}",
            total_mimic_loss / num_iters,
            args.curr_iter,
        )
        args.writer.add_scalar(f"Acc/val_{dataset_name}", accuracy, args.curr_iter)

    return accuracy


def load(load_path, encoder, model, device):
    """
    Loads the support-set Transformer model, and possibly the encoder, stored
    at load_path
    """
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    if "encoder" in state_dict:
        encoder.load_state_dict(state_dict["encoder"])


def load_encoder(load_path, encoder, device):
    """
    Loads ONLY the encoder stored at load_path
    """
    state_dict = torch.load(load_path, map_location=device)
    encoder.load_state_dict(state_dict["encoder"])


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Load encoder
    encoder, latent_dim = get_encoder(args)
    args.latent_dim = latent_dim

    # Load support-set Transformer model
    if args.use_pmf:
        from model.pmf import get_model
    else:
        from model.mimic import get_model
    model = get_model(args)

    # Load from checkpoint
    if args.load_path != "":
        load(args.load_path, encoder, model, args.device)
    if args.load_encoder != "":
        load_encoder(args.load_encoder, encoder, args.device)

    # Dataset setup
    if args.eval_on_val:
        _, test_loaders, test_names = get_loaders(args, train_and_val=True, test=False)
    else:
        test_loaders, test_names = get_loaders(args, train_and_val=False, test=True)

    # Normalization work
    assert not (
        args.eval_standardize and args.eval_standardize_train
    ), "Cannot specify both eval_standardize and eval_standardize_train"
    if args.eval_standardize_train:
        assert args.dataset == "hoi", "Dataset statistics computed only over hoi"
        settings_dict = torch.load(
            args.eval_standardize_train, map_location=args.device
        )
        args.mean = torch.tensor(settings_dict["mean"], device=args.device)
        args.std = torch.tensor(settings_dict["std"], device=args.device)

    # Logging and saving setup
    save_name = (
        f"{args.dataset}"
        f"_{args.baseline}"
        f"_{args.arch}"
        f"_{os.path.split(args.load_path)[-1]}"
        f"_dropout{args.eval_dropout_support}"
        f"_noise{args.eval_label_noise}"
        f"_k{args.k}"
        f"_stdize{args.eval_standardize}"
        f"_val{args.eval_on_val}"
        f"_seed{args.seed}"
    )
    results_path = os.path.join(args.results_dir, save_name)
    args.writer = None

    # Evaluation
    accuracies = defaultdict(list)
    for test_name, test_loader in zip(test_names, test_loaders):
        log.info(f"Evaluating on dataset {test_name}")
        accuracy = eval_(
            model,
            encoder,
            args,
            test_loader,
            test_name,
            eval_dropout=args.eval_dropout_support,
            eval_label_noise=args.eval_label_noise,
            baseline=args.baseline,
        )
        accuracies[test_name].append(accuracy)

    # Save
    pd.DataFrame(accuracies).to_csv(results_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
