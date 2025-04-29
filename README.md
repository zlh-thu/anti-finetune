# Anti-FT: Towards Practical Deep Leakage from Gradients

This repository is the official implementation of our ICIP 2025 submission Anti-FT: Towards Practical Deep Leakage from Gradients. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Make sure the directory follows:
```File Tree
stealingverification
├── data
│   ├── cifar10
│   │   ├── train
│   │   ├── test
│   │   └── finetune
│   ├── sub-imagenet-20
│   │   ├── train
│   │   ├── test
│   │   └── finetune
│   ├── sub-imagenet-20
│   ├── subimage_8_client0
│   └── ...
├── network
│   
├── save
│   
├── inv_script
│   
├── model
│   
├── inversefed
|
```


## Dataset Preparation
Make sure the directory ``data`` follows:
```File Tree
data
├── cifar10
│   ├── train
│   ├── test
│   └── finetune
├── sub-imagenet-20
│   ├── train
│   ├── test
│   └── finetune
│ 
├── cifar10_8_client0
├── subimage_8_client0
│   └── ...
```


>📋  Data Download Link:  
>[data](https://www.dropbox.com/sh/kcgql440dp5m1v1/AADqRigYXXlvZBVo7nKdoWqCa?dl=0)


## Model Preparation
>📋  CheckPoint Download Link:  
>[model](https://www.dropbox.com/sh/21310c3moqtneqp/AACsRdgUTxChrLoZENOz2TgMa?dl=0)



## Anti-Finetune 
Attack target client with Anti-Finetune. The results will be saved in ``save``.

CIFAR-10:
```Attack
python federated_main.py --epochs=312500 --local_bs=4   --attack_num=100 --gpu=0 --exp_id=1 --load_data_from_dir=./data --changing_round=156250 --mood=avg_loss --attack=anti_finetune --model=resnet18 --leakage_attack --anti_finetune_ep=5 --max_iterations=4000
python federated_main.py --epochs=4883   --local_bs=128 --attack_num=100 --gpu=0 --exp_id=1 --load_data_from_dir=./data --changing_round=0      --mood=avg_loss --attack=anti_finetune --model=resnet18 --leakage_attack --anti_finetune_ep=5 --max_iterations=4000 --resume=./model/ckpt/cifar10_global_epoch4883.pt
```

ImageNet:
```Attack
python federated_main.py --epochs=312500 --dataset=imagenet --num_classes=20 --local_bs=4  --attack_num=100 --gpu=0 --exp_id=1 --load_data_from_dir=./data --changing_round=0 --mood=avg_loss --attack=anti_finetune --model=resnet18_backbone --leakage_attack --anti_finetune_ep=5 --max_iterations=24000 --pretrained --frozen
```


