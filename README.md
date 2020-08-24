# Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision

Code release for "Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision" (ECCV 2020) ([PDF])(https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540528.pdf)

[Supplementary Material](https://github.com/GeraldHan/TRN/blob/master/0895-Supp.pdf)

```
@article{haninterpretable,
  title={Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision},
  author={Han, Xinzhe and Wang, Shuhui and Su, Chi and Zhang, Weigang and Huang, Qingming and Tian, Qi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}  
}
```

This repo contains the experiments on both VQA v2 and CLEVR dataset. The implementation details are listed in `./exp_vqa` and `./exp_clevr`.

## Dependencies
- python 3.6
- pytorch >= 1.1
- spaCy 2.2
- tqdm 4.42
- h5py 2.10
- numpy 1.18

## Acknowledgements
Baseline models are referred to https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch

Data process for CLEVR are referred to [NS-VQA](https://github.com/kexinyi/ns-vqa.git)
