import torch
from torchvision.models.resnet import resnet50

from .clip.clip import load
from .resnet12 import resnet12


def get_encoder(args):
    if args.arch == "custom":
        assert (
            args.train_encoder
        ), "For custom backbone, --train_encoder must be set at train and test time"

        latent_dim = 128
        in_channels = 1 if args.dataset == "logo" or args.dataset == "classic" else 3

        encoder = resnet12(out_dim=latent_dim, inplanes=in_channels).to(
            device=args.device
        )
    elif args.arch == "clip":
        DOWNLOAD_ROOT = ".cache/clip"

        clip, _, _ = load("RN50", device=args.device, download_root=DOWNLOAD_ROOT)
        encoder = clip.visual

        latent_dim = 1024
    elif args.arch == "dino_vit_base":
        # Modified from https://github.com/hushell/pmf_cvpr22/blob/main/models/__init__.py
        from . import dino_vit as vit

        encoder = vit.__dict__["vit_base"](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )

        encoder.load_state_dict(state_dict, strict=True)
        print("Pretrained weights found at {}".format(url))

        encoder = encoder.to(device=args.device)
        latent_dim = 768
    else:
        raise NotImplementedError(f"{args.arch} is not a valid encoder backbone")

    # Freeze weights if encoder is not trainable
    if not args.train_encoder:
        for _, param in encoder.named_parameters():
            param.requires_grad_(False)

    return encoder, latent_dim


def encode(encoder, x_support, x_query, use_grad=False):
    B, N_sup = x_support.shape[:2]
    N_targ = x_query.shape[1]

    # Concatenate supports and queries and embed them all at the same time
    input_ = torch.cat([x_support, x_query], dim=1).reshape((-1, *x_support.shape[2:]))
    if use_grad:
        all_emb = encoder(input_)
    else:
        with torch.no_grad():
            all_emb = encoder(input_)

    all_emb = all_emb.reshape((B, N_sup + N_targ, -1))

    x_support_emb = all_emb[:, :N_sup]
    x_query_emb = all_emb[:, N_sup:]

    assert x_support_emb.shape[:2] == x_support.shape[:2]
    assert x_query_emb.shape[:2] == x_query.shape[:2]

    return x_support_emb, x_query_emb
