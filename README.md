# Cross-Image Context Matters for Bongard Problems

This is the official implementation of "Cross-Image Context Matters for Bongard Problems." [[Paper](https://arxiv.org/abs/2309.03468)] [[Project Page](https://nikhilraghuraman.com/projects/bongard.html)]

## Overview
We introduce two approaches for incorporating cross-image context when solving Bongard problems: support-set standardization and support-set Transformers. We attain state-of-the-art performance on two Bongard datasets. We additionally attain strong performance on a third Bongard dataset where comparison with prior methods is not possible due to different evaluation.
- State-of-the-art on [Bongard-HOI](https://github.com/NVlabs/Bongard-HOI)
- State-of-the-art on [Bongard-LOGO](https://github.com/NVlabs/Bongard-LOGO)
- Strong performance on [Bongard-Classic](https://github.com/XinyuYun/bongard-problems)

## Setup
To install all needed packages, run the following command:
```
pip install -r requirements.txt
```
For information on dataset setup, please see `datasets/README.md`.

## Model Checkpoints
We include model checkpoints of our best-performing model, SVM-Mimic, on Bongard-HOI and Bongard-LOGO in [this Google Drive folder](https://drive.google.com/drive/folders/1WyA5QN3GDqa_HOCC2NoB9-2gdYcmp6db?usp=sharing). Please see the commands below to evaluate them. These models attain the following accuracy on Bongard-HOI and Bongard-LOGO test splits:

`hoi_checkpoint.pt`

| unseen object, unseen act | unseen object, seen act | seen object, unseen act | seen object, seen act | Avg |
| --- | --- | --- | --- | --- |
| 69.70 | 71.04 | 78.43 | 71.30 | 72.62 |

`logo_checkpoint.pt`

| freeform | combinatorial | basic shape | novel | Avg |
| --- | --- | --- | --- | --- |
| 73.33 | 69.75 | 83.85 | 74.53 | 75.37 |

## Important Command-Line Arguments
All command-line arguments for both training and testing are documented in `utils/args.py`. We briefly mention the most important ones here:
- `--dataset` can be used to specify which dataset to train/evaluate on. We currently support one of `hoi`, `logo` or `classic`. You can add more by modifying the `get_loaders()` function in `datasets/__init__.py`.
- `--arch` can be used to specify the encoder backbone. We currently support one of `custom`, `clip`, or `dino_vit_base`. You can add more by modifying the `get_encoder()` function in `model/encoder.py`.
- `--train_encoder` specifies whether to keep the encoder frozen or not. To train with the `custom` encoder or with the PMF objective as explained in the paper, `--train_encoder` must be set.
- `--balance_dataset` specifies whether to use our cleaned Bongard-HOI dataset annotations, as explained in the paper. It is only useful to set this argument at train time as we have no cleaned annotations on the test set.
- `--baseline` can be used to run one of the baselines in the paper (one of `SVM`, `PROTOTYPE`, or `KNN`). This is only used at test time.

## Logging
We log various metrics (accuracy, loss, etc.) to Tensorboard, storing runs data in the `runs/` directory. All code for this logic is in `eval.py` and `main.py`. Feel free to remove this logging or change the code to log using a different platform.

## Replicating Results

### Bongard-HOI

To train SVM-Mimic with our hyperparameter choices, run the following command:
```
python3 main.py /path/to/hoi/dataset --batch_size 16 --lr 5e-5 --use_augs --arch clip --balance_dataset --train_steps 10000 --use_scheduler --dropout_support_prob 1.0 --label_noise_support_prob 0.25
```

To train Prototype-Mimic, run the following command:
```
python3 main.py /path/to/hoi/dataset --batch_size 16 --lr 5e-5 --use_augs --arch clip --balance_dataset --train_steps 10000 --use_scheduler --dropout_support_prob 1.0 --label_noise_support_prob 0.25 --mimic_kind PROTOTYPE
```

To test SVM-Mimic, run the following command:
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --load_path checkpoints/name_of_hoi_checkpoint
```

To test Prototype-Mimic, run the following command:
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --load_path checkpoints/name_of_hoi_checkpoint --mimic_kind PROTOTYPE
```

To test SVM, Prototype, and KNN baselines with and without standardization, run the following:
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline SVM
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline SVM --eval_standardize
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline PROTOTYPE
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline PROTOTYPE --eval_standardize
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline KNN --k 3
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline KNN --k 3 --eval_standardize
```

To test SVM, Prototype, and KNN baselines with different forms of normalization, run the following:
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline SVM --eval_normalize_l2
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline SVM --eval_standardize_train checkpoints/dataset_statistics.pt
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline PROTOTYPE --eval_standardize_train checkpoints/dataset_statistics.pt
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --baseline KNN --k 3 --eval_standardize_train checkpoints/dataset_statistics.pt
```

To train PMF, run the following:
```
python3 main.py /path/to/hoi/dataset --batch_size 4 --lr 5e-7 --use_augs --arch clip --balance_dataset --train_steps 40000 --use_scheduler --dropout_support_prob 1.0 --use_pmf --train_encoder 
```

To train PMF + SVM-Mimic, run the following (for Prototype-Mimic, append `--mimic_kind PROTOTYPE`):
```
python3 main.py /path/to/hoi/dataset --batch_size 16 --lr 5e-5 --use_augs --arch clip --balance_dataset --train_steps 10000 --use_scheduler --dropout_support_prob 1.0 --label_noise_support_prob 0.25 --load_path checkpoints/name_of_pmf_checkpoint
```

To test PMF + SVM-Mimic, note that it is necessary to load both the PMF encoder (with `--load_encoder`) and the SVM-Mimic Transformer model (with `--load_path`). You can use the following command:
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --load_encoder checkpoints/name_of_pmf_checkpoint --load_path checkpoints/name_of_hoi_svm_mimic_checkpoint
```

To test PMF backbones with various baselines, run the following (append `--eval_standardize` to standardize, and change `SVM` to the baseline of choice). Note that to obtain the PMF + standardize
results, it is sufficient to run with baseline `PROTOTYPE` and set `--eval_standardize`.
```
python3 test.py /path/to/hoi/dataset --batch_size 1 --arch clip --load_path checkpoints/name_of_pmf_checkpoint --use_pmf --train_encoder --baseline SVM
```

### Bongard-LOGO

Bongard-LOGO commands are similar. E.g., to train SVM-Mimic on Bongard-LOGO with our hyperparameter choices, run the following:
```
python3 main.py /path/to/logo/dataset --dataset logo --arch custom --train_encoder --batch_size 2 --train_steps 500000 --use_scheduler --dropout_support_prob 1.0 --lr 5e-5 --temperature 0.1 --resolution 512 --use_augs --weight_decay 0.0001
```

To test SVM-Mimic on Bongard-LOGO, run the following (note that `--train_encoder` is set even at test time):
```
python3 test.py /path/to/logo/dataset --dataset logo --arch custom --train_encoder --resolution 512 --temperature 0.1 --batch_size 1 --load_path checkpoints/name_of_logo_checkpoint
```

### Bongard-Classic

We have no training pipeline on Bongard-Classic, only evaluation pipelines involving encoders and support-set Transformers pre-trained on Bongard-LOGO. To evaluate SVM-Mimic (trained on Bongard-LOGO) on Bongard-Classic, run the following command:
```
python3 test.py /path/to/classic/dataset --dataset classic --arch custom --train_encoder --batch_size 1 --resolution 512 --temperature 0.1 --load_path checkpoints/name_of_logo_checkpoint
```
To evaluate with Prototype-Mimic or one of the baselines, append `--mimic_kind PROTOTYPE` or `--baseline BASELINE_NAME` to the command. See the Bongard-HOI commands for examples.

## Bibtex
If you use our project to inform your work, please consider citing us. Thank you!
```
@article{raghuraman2023cross,
  title={Cross-Image Context Matters for Bongard Problems}, 
  author={Nikhil Raghuraman and Adam W. Harley and Leonidas Guibas},
  year={2023},
  journal={arXiv:1804.04452},
}
```
