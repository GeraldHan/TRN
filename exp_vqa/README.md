# TRN experiments on VQA v2.0
Baseline models are referred to https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch

## Dependencies
- python 3.6
- pytorch >= 1.3
- spaCy 2.2
- tqdm 4.42
- h5py 2.10
- numpy 1.18


## Pipeline to preprocess data
1. Download grounded features from the [repo](https://github.com/peteanderson80/bottom-up-attention) of paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
```
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
```
process features use `process/features.py` and output `genome-trainval_36.h5` and  `genome-test.h5`(optional). Change the paths to your own save folders.

2.  Download glove pretrained 300d word vectors and dictionary
```
wget https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch/blob/master/data/glove6b_init_300d.npy

wget https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch/blob/master/data/dictionary.pkl
```
3. Preprocess VQA v2.0 questions. Change the paths to your own save folders.
```
bash process/question.sh
```

Before training, make sure your have following files:
- vocab.json
- train_questions.pt
- genome-trainval_36.h5
- val_questions.pt (optional)
- test_questions.pt (optional)
- trainval_quetions.pt (optional)
- genome-test.h5 (optional)

## Pipeline for Train models
1. Change the paths in `train.py` to your own save folders.

2. Change `from model_* import Net` to choose a model.
- `model_updn.py`: Baseline for UpDn
- `model_ban.py`: Baseline for BAN
- `model_timeupdn.py`: UpDn + TRN
- `model_timeban.py`: BAN + TRN

3. Train with `bash train_vqa.sh`. Change the paths to your own directions. `--val` is optional for validate every epoch, `--restore` is for fine-tune from a checkpoint.

## Validate and Test
1. Change the paths in `validate.py` to your own save folders.

2. Change `from model_* import Net` to choose a model.

3. `bash test.sh` can output a `*.json` for online test and validate on validation split.

Result files that can be evaluated online is provided in `./result`

## Visualization
Following instructions in `visualize.ipynb`. Change `model_*.py` you can visualize other examples. You should return `z_seq` instead of `loss_time`.