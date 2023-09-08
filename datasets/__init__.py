import numpy as np
import torch.utils.data

from .classic_dataset import BongardClassicDataset
from .hoi_dataset import HoiDataset
from .logo_dataset import LogoDataset


def get_loaders(args, train_and_val=True, test=False):
    assert train_and_val or test

    if args.dataset == "hoi":
        eval_names = [
            "unseen_obj_unseen_act",
            "unseen_obj_seen_act",
            "seen_obj_unseen_act",
            "seen_obj_seen_act",
        ]

        if train_and_val:
            train_dataset = HoiDataset(
                args.data,
                dset="train",
                balance_dataset=args.balance_dataset,
                use_augs=args.use_augs,
                img_size=args.resolution,
                backbone=args.arch,
            )

            val_datasets = []
            for dataset_name in eval_names:
                val_dataset = HoiDataset(
                    args.data,
                    data_split=dataset_name,
                    dset="val",
                    balance_dataset=args.balance_dataset,
                    img_size=args.resolution,
                    use_augs=False,
                    backbone=args.arch,
                )
                val_datasets.append(val_dataset)
        else:
            test_datasets = []
            for dataset_name in eval_names:
                test_dataset = HoiDataset(
                    args.data,
                    data_split=dataset_name,
                    dset="test",
                    balance_dataset=False,  # only available for train/val
                    img_size=args.resolution,
                    use_augs=False,
                    backbone=args.arch,
                )
                test_datasets.append(test_dataset)
    elif args.dataset == "logo":
        if train_and_val:
            eval_names = ["val"]
            train_dataset = LogoDataset(
                args.data,
                dset="train",
                img_size=args.resolution,
                use_augs=args.use_augs,
            )

            val_datasets = []
            for dataset_name in eval_names:
                val_dataset = LogoDataset(
                    args.data,
                    dset="val",
                    img_size=args.resolution,
                    use_augs=False,
                )
                val_datasets.append(val_dataset)
        else:
            eval_names = ["test_ff", "test_hd_comb", "test_bd", "test_hd_novel"]
            test_datasets = []
            for dataset_name in eval_names:
                test_dataset = LogoDataset(
                    args.data,
                    dset=dataset_name,
                    img_size=args.resolution,
                    use_augs=False,
                )
                test_datasets.append(test_dataset)
    elif args.dataset == "classic":
        if train_and_val:
            raise ValueError("No train or val set on Bongard-Classic dataset")
        else:
            eval_names = ["test"]
            test_datasets = []
            for dataset_name in eval_names:
                test_dataset = BongardClassicDataset(
                    args.data, img_size=args.resolution
                )
                test_datasets.append(test_dataset)
    else:
        raise NotImplementedError(
            f"Attempting to use non-existent dataset {args.dataset}"
        )

    if train_and_val:
        if args.train_subset < 1.0:
            rng = np.random.default_rng(0)
            train_inds = rng.choice(
                len(train_dataset),
                int(len(train_dataset) * args.train_subset),
                replace=False,
            )
            train_dataset = torch.utils.data.Subset(train_dataset, train_inds)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_loaders = []
        for val_dataset in val_datasets:
            if args.val_subset < 1.0:
                rng = np.random.default_rng(0)
                val_inds = rng.choice(
                    len(val_dataset),
                    int(len(val_dataset) * args.val_subset),
                    replace=False,
                )
                val_dataset = torch.utils.data.Subset(val_dataset, val_inds)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
            val_loaders.append(val_loader)

        return train_loader, val_loaders, eval_names
    else:
        test_loaders = []
        for test_dataset in test_datasets:
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
            test_loaders.append(test_loader)

        return test_loaders, eval_names
