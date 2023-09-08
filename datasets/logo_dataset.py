# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT
#
# Adapted from https://github.com/NVlabs/Bongard-LOGO/blob/master/Bongard-LOGO_Baselines/datasets/shape_bongard_v2.py
import json
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUM_PER_CLASS = 7


class LogoDataset(Dataset):
    def __init__(
        self,
        data_root,
        dset="train",
        img_size=512,
        use_augs=False,
    ):
        self.dset = dset

        split_file = os.path.join(data_root, "ShapeBongard_V2_split.json")
        split = json.load(open(split_file, "r"))

        self.tasks = sorted(split[dset])
        self.n_tasks = len(self.tasks)
        print("found %d tasks in dset %s" % (self.n_tasks, dset))

        task_paths = [
            os.path.join(data_root, task.split("_")[0], "images", task)
            for task in self.tasks
        ]

        self.task_paths = task_paths

        norm_params = {"mean": [0.5], "std": [0.5]}  # grey-scale to [-1, 1]
        normalize = transforms.Normalize(**norm_params)

        if use_augs:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.75, 1.2)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.task_paths)

    def __getitem__(self, index):
        task_path = self.task_paths[index]

        neg_imgs = []
        for idx in range(NUM_PER_CLASS):
            file_path = "%s/0/%d.png" % (task_path, idx)
            img = Image.open(file_path).convert("L")
            neg_imgs.append(self.transform(img))
        pos_imgs = []
        for idx in range(NUM_PER_CLASS):
            file_path = "%s/1/%d.png" % (task_path, idx)
            img = Image.open(file_path).convert("L")
            pos_imgs.append(self.transform(img))

        neg_imgs = torch.stack(neg_imgs, dim=0)
        pos_imgs = torch.stack(pos_imgs, dim=0)

        if self.dset == "train":
            perm = np.random.permutation(NUM_PER_CLASS)
            pos_imgs = pos_imgs[perm]
            perm = np.random.permutation(NUM_PER_CLASS)
            neg_imgs = neg_imgs[perm]

        pos_support = pos_imgs[:-1]
        neg_support = neg_imgs[:-1]
        pos_query = pos_imgs[-1]
        neg_query = neg_imgs[-1]

        x_support = torch.cat([pos_support, neg_support], dim=0)
        x_query = torch.stack([pos_query, neg_query])
        y_support = torch.cat(
            [torch.ones(len(pos_support)), torch.zeros(len(neg_support))], dim=0
        )
        y_query = torch.tensor([1, 0])

        return x_support, x_query, y_support.long(), y_query.long()
