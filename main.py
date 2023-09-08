import datetime
import os.path

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import get_loaders
from test import eval_
from model.encoder import get_encoder, encode
from utils.args import get_args
from utils.baselines import Baseline
from utils.tools import (
    SimplePool,
    log,
)


N_POOL = 100


def save_state(model, encoder, save_path, curr_iter, optimizer, scheduler):
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curr_iter": curr_iter,
        "scheduler": scheduler,
    }
    if args.train_encoder or args.load_path:
        save_dict["encoder"] = encoder.state_dict()

    torch.save(save_dict, save_path)


def train(model, encoder, args, train_loader, val_loaders, optimizer, scheduler=None):
    model.train()
    if args.train_encoder:
        encoder.train()
    else:
        encoder.eval()

    args.curr_iter = 0
    epoch = 0
    # To enable smoother Tensorboard plots, we pool various metrics over some
    # time window in the past (specified by N_POOL). We plot both the pooled
    # and the unpooled forms of the metrics.
    enc_loss_pool_t = SimplePool(N_POOL, version="np")
    mimic_loss_pool_t = SimplePool(N_POOL, version="np")
    acc_pool_t = SimplePool(N_POOL, version="np")
    while args.curr_iter < args.train_steps:
        epoch_loss = 0
        epoch_iters = 0
        epoch += 1

        epoch_total = 0
        epoch_correct = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

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

            x_support_emb, x_query_emb = encode(
                encoder,
                x_support,
                x_query,
                use_grad=args.train_encoder,
            )
            acc, _, enc_loss, mimic_loss = model(
                x_support_emb, x_query_emb, y_support, y_query
            )
            loss = mimic_loss + enc_loss
            loss.backward()

            optimizer.step()
            if args.use_scheduler:
                scheduler.step()

            epoch_loss += loss.item()

            epoch_total += 1
            epoch_correct += torch.mean(acc).item()

            acc = torch.mean(acc)

            # Add unpooled versions
            args.writer.add_scalar("Enc_loss/train", enc_loss.item(), args.curr_iter)
            args.writer.add_scalar(
                "Mimic_loss/train", mimic_loss.item(), args.curr_iter
            )
            args.writer.add_scalar("Acc/train", acc.item(), args.curr_iter)
            if args.use_scheduler:
                args.writer.add_scalar(
                    "LR/lr", scheduler.get_last_lr()[0], args.curr_iter
                )

            # Add pooled versions
            enc_loss_pool_t.update([enc_loss.detach().cpu().numpy()])
            args.writer.add_scalar(
                "Enc_loss_pooled/train", enc_loss_pool_t.mean(), args.curr_iter
            )
            mimic_loss_pool_t.update([mimic_loss.detach().cpu().numpy()])
            args.writer.add_scalar(
                "Mimic_loss_pooled/train", mimic_loss_pool_t.mean(), args.curr_iter
            )
            acc_pool_t.update([acc.detach().cpu().numpy()])
            args.writer.add_scalar(
                "Acc_pooled/train", acc_pool_t.mean(), args.curr_iter
            )

            epoch_iters += 1
            args.curr_iter += 1

            if args.curr_iter >= args.train_steps:
                break
            elif args.curr_iter % args.eval_every == 0:
                for dataset_name, val_loader in zip(args.data_split, val_loaders):
                    eval_(
                        model,
                        encoder,
                        args,
                        val_loader,
                        dataset_name,
                        baseline=Baseline.NONE,
                    )
                    model.train()
                    if args.train_encoder:
                        encoder.train()
                    else:
                        encoder.eval()

            if args.curr_iter % args.save_every == 0:
                save_path = os.path.join(
                    args.save_dir, f"{args.save_name}_iter{args.curr_iter}"
                )
                save_state(
                    model,
                    encoder,
                    save_path,
                    args.curr_iter,
                    optimizer,
                    scheduler if args.use_scheduler else None,
                )

        log.info(f"Loss after epoch {epoch}: {epoch_loss / epoch_iters}")
        log.info(f"Acc after epoch {epoch}: {epoch_correct / epoch_total}")

    save_path = os.path.join(args.save_dir, args.save_name)
    save_state(
        model,
        encoder,
        save_path,
        args.curr_iter,
        optimizer,
        scheduler if args.use_scheduler else None,
    )

    # Final eval
    for dataset_name, val_loader in zip(args.data_split, val_loaders):
        eval_(model, encoder, args, val_loader, dataset_name, baseline=Baseline.NONE)


def load(load_path, encoder):
    """
    Loads pretrained weights specified at load_path into the encoder only (not the Transformer model)
    """
    state_dict = torch.load(load_path)
    if "encoder" in state_dict:
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
    if args.load_path != "":
        load(args.load_path, encoder)

    # Set up optimizer and scheduler
    parameters = model.parameters()
    if args.train_encoder:
        parameters = list(parameters) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters, weight_decay=args.weight_decay, lr=args.lr
    )
    if args.use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            args.lr,
            total_steps=args.train_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
    else:
        scheduler = None

    # Dataset setup
    train_loader, val_loaders, val_names = get_loaders(
        args, train_and_val=True, test=False
    )
    args.data_split = val_names

    # Logging and saving setup
    save_name = (
        f"{args.dataset}"
        f"_lr{args.lr}"
        f"_bs{args.batch_size}"
        f"_temp{args.temperature}"
        f"_arch{args.arch}"
        f"_scheduler{args.use_scheduler}"
        f"_train_enc{args.train_encoder}"
        f"_noise{args.label_noise_support_prob}"
        f"_kind{args.mimic_kind}"
        f"_{datetime.datetime.now().strftime('%m-%d_%H:%M:%S')}"
    )
    if args.id:  # For easier disambiguation of identical runs
        save_name += args.id
    writer = SummaryWriter(log_dir=os.path.join(args.runs_dir, save_name))
    args.writer = writer
    args.save_name = save_name

    train(
        model, encoder, args, train_loader, val_loaders, optimizer, scheduler=scheduler
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
