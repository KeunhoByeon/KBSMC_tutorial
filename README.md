# KBSMC_tutorial

### Install Enviroments
```
conda update --all
conda create -n KBSMC_tutorial python=3.8
conda activate KBSMC_tutorial
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# pytorch download link
https://pytorch.org/get-started/locally/
```

### Install Openslide
```
openslide-win64.zip 직접 다운로드 및 압축 해제(권장)
https://drive.google.com/uc?export=download&id=1AJFGTg73m5Bi4G3EBPQttVbt8iL0ujFa

다운로드 명령어 및 압축해제 명령어
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AJFGTg73m5Bi4G3EBPQttVbt8iL0ujFa' -O openslide-win64.zip
unzip ./openslide-win64.zip 
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

### 3. Train and Test Segmentor

```
4-1. train_segmentor.py
4-2. test_segmentor.py
```
