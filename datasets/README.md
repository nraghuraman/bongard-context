# Dataset Setup Instructions

## Bongard-HOI Dataset
1. First, follow the official instructions to download the Bongard-HOI dataset to any location of your choice: https://github.com/NVlabs/Bongard-HOI. If downloading the images from Zenodo is too slow, try downloading the HAKE images directly.
2. Ensure that you have the following directory structure (you will need to add the `cache/` subdirectory). This directory tree was adapted from the HAKE download instructions.
   ```
   /path/to/hoi/dataset/
    ├── cache
    ├── hake_images_20190730
    │   └── xxx.jpg
    ├── hake_images_20200614
    │   └── xxx.jpg
    ├── hcvrd
    │   └── xxx.jpg
    ├── hico_20160224_det
    │   └── images
    │       ├── test2015
    │       │   └── xxx.jpg
    │       └── train2015
    │           └── xxx.jpg
    ├── openimages
    │   └── xxx.jpg
    ├── pic
    │   └── xxx.jpg
    └── vcoco
        ├── train2014
        |    └── xxx.jpg
        └── val2014
            └── xxx.jpg
   ```
3. As explained in the Bongard-HOI download instructions, download the official Bongard-HOI annotations and extract them into `cache/`.
4. To use our cleaned/balanced Bongard-HOI annotations (see paper for details), copy the `bongard_hoi_clean/` directory to `cache/`. In the code, you can enable use of these cleaned annotations by setting the `--balance_dataset` flag. Note that we only have cleaned annotations for the train and validation sets, not the test set.
5. Verify that your `cache/` directory has the following structure:
   ```
   cache/
   ├── bongard_hoi_clean
   |   └── bongard_hoi_train.json
   |   └── ...
   ├── bongard_hoi_release
   |   └── bongard_hoi_train.json
   |   └── ...
   ```

## Bongard-LOGO Dataset
Follow the official instructions to download Bongard-LOGO to any location of your choosing: https://github.com/NVlabs/Bongard-LOGO

## Bongard-Classic Dataset
Clone this repository to any location of your choosing to download Bongard-Classic: https://github.com/XinyuYun/bongard-problems

## Adding Your Own Datasets
The main entry point to the datasets in the code is the `get_loaders()` function in `__init__.py`. You can add a custom dataset by creating a new `torch.util.data.Dataset` in this directory and adding logic for the loader in `get_loaders()`. Then, this new dataset can be easily specified on the command line with `--dataset NAME_OF_DATASET`.
