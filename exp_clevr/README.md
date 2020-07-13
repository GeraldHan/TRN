# TRN experiments on CLEVR
Pytorch implementation for TRN on the [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/). 

Baseline models are referred to https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch

## Dependencies
- python 3.6
- pytorch >= 1.3
- spaCy 2.2
- tqdm 4.42
- h5py 2.10
- numpy 1.18

## Pipeline to preprocess data
1. Extract grounded features according to [NS-VQA](https://github.com/kexinyi/ns-vqa.git) of paper **[Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338)**

    Following the instructions in NS-VQA for Mask-RCNN and object detection. Generate object files at `{repo_root}/data/attr_net/objects/clevr_val_objs_pretrained.json` and `{repo_root}/data/attr_net/objects/clevr_train_objs_pretrained.json`.

    Copy `save_vec.py` to `ns-vqa/scene_parse/attr_net/tools/`. Change * to `train` or `val` and generate vision feature files `train_features.h5` and `val_features.h5`.
    ```
    cd scene_parse/attr_net

     python tools/save_vec.py \
    --run_dir ../../data/attr_net/results \
    --dataset clevr \
    --load_checkpoint_path ../../data/pretrained/attribute_net.pt \
    --clevr_val_ann_path ../../data/attr_net/objects/clevr_*_objs_pretrained.json \
    --output_path ../../data/attr_net/results/clevr_*_feature.h5 \
    --split * \
    --clevr_val_img_dir ../../data/raw/CLEVR_v1.0/images/*
    ```

2. Preprocess VQA v2.0 questions. Change the paths to your own save folders.
    ```
    cd process

    bash process_q.sh
    ```

Before training, make sure your have following files:
- input/vocab.json
- input/train_questions.pt
- input/val_questions.pt
- input/train_features.h5
- input/val_features.h5
- input_human/vocab.json
- input_human/train_questions.pt
- input_huamn/genome-trainval_36.h5
- input_human/val_questions.pt

## Train CLEVR
1. Change the paths in `train.py` to your own save folders. You can directly use `input_human/vocab.json` for CLEVR.

2. Change `from model_* import Net` to choose a model.
- `model_updn.py`: Baseline for UpDn
- `model_ban.py`: Baseline for BAN
- `model_dfaf.py`: Baseline for DFAF
- `model_updntime.py`: UpDn + TRN
- `model_timeban.py`: BAN + TRN
- `model_dfaftime.py`: DFAF + TRN

3. Train with `bash train_clevr.sh`. Change the paths to your own directions. `--val` is optional for validate every epoch, `--restore` is for fine-tune from a checkpoint.

## Train CLEVR-Humans
1. Change the paths in `train.py` to your own save folders.

2. Pre-train with `bash train_clevr.sh`. Change the paths to your own directions. `--val` is optional for validate every epoch, `--restore` is for fine-tune from a checkpoint.

3. Fine-tune model with `bash trian_human.sh`. Change the paths to your own directions. 

## Visualization
Following instructions in `visualize.ipynb`. Change `model_*.py` you can visualize other examples. You should return `z_seq` instead of `loss_time`.