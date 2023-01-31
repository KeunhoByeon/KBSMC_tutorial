# KBSMC_tutorial

### 0. Install Enviroments
```
conda update --all
conda create -n KBSMC_tutorial python=3.8
conda activate KBSMC_tutorial
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# pytorch download link
https://pytorch.org/get-started/locally/
```

### 1. Make Patches

```
# 1-1. make_mask.py
# 1-2. make_patches_simple.py
1-3. make_patches.py
# 1-4. make_patches_multiprocessing.py
```

### 2. Dataloader

```
2-1. dataloader.py
```

### 3. Train and Test Classifier

```
3-1. train_classifier.py
3-2. test_classifier.py
```
