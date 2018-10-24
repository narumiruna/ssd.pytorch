# BDD

1. Copy data/coco_labels.txt to ~/data/coco/coco_labels.txt

2. Make sure your directory structure of bdd dataset:

```
.
├── bdd100k
│   ├── images
│   │   ├── 100k
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   └── 10k
│   │       ├── test
│   │       ├── train
│   │       └── val
│   └── labels
│       ├── train
│       └── val
```

3. Train ssd

```sh
python train.py --dataset BDD --dataset_root ~/datasets/bdd/bdd100k --lr 1e-4 --cuda true
```
