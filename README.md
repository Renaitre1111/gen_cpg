# Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
Yaxin Hou, Bo Han, Yuheng Jia, Hui Liu, Junhui Hou, Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning, Advances in Neural Information Processing Systems, 2nd-7th December, San Diego, 2025.

This is an official [PyTorch](http://pytorch.org) implementation for **Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning**.

## Introduction
This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning).

What I've done is just adding our CPG algorithm in `semilearn/imb_algorithms/cpg`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run
For example, on CIFAR-10-LT with long-tailed labeled data ($\gamma_l=100$) and arbitrary unlabeled data($\gamma_u=100$)

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1.yaml"
```

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-0/gen_cpg/203-fixmatch_gen_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_0.yaml"
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `/saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/log.txt` will tell you the best accuracy. 

For example,
```
[2025-04-20 13:35:17,784 INFO] model saved: ./saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/latest_model.pth
[2025-04-20 13:35:17,815 INFO] Model result - eval/best_acc : 0.8233
[2025-04-20 13:35:17,816 INFO] Model result - eval/best_it : 244735
```

## Results

## simsiam training
python simsiam/finetune.py --data ./data/ --lb_idx_path simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy --ulb_idx_path simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/ulb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy --arch resnet18 --lr 1e-3 --epochs 250 --seed 0 --save_freq 30 --save_dir simsiam/saved_model/ --fix_pred_lr

# linear eval
python simsiam/linear_eval.py --data ./data/ --pretrained simsiam/saved_model/checkpoint_final.pth --lb_idx_path simsiam/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy --arch resnet18 --lr 0.1 --epochs 100 --seed 0 --wd 0