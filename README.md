# Learn Fashion Compatibility with Diagnose Attention
## Task Description

## Experiment Results


## Dependency:

* pytorch==0.4.0
* sklearn
* scipy
* pillow


## Usage

1. Download dataset from [Google Drive](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0), which refers to <https://github.com/xthan/polyvore-dataset>

2. Train

Revise the ``root_dir`` for ``train_dataset``, ``val_dataset``, ``test_dataset``, which is the directory stores the source images.

```
cd relations
python train_relation_vse_type.py
tail -f log_train_relation_vse_type.log 
```

3. Evaluate
```
python evaluate_relation_vse_type.py
```
