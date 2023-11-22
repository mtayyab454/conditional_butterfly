# Diffusion based image super-resolution using knowlege of measurement system

## Getting started 

### 1) Install dependencies
Install conda and create conda environment from environment.yml
```
conda env create -f environment.yml
```

### 2) Clone the repository

```
git clone https://https://github.com/mtayyab454/conditional_butterfly

cd conditional_butterfly
```

### 3) Download dataset
```
wget https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset/blob/main/data/train-00000-of-00001.parquet
```

### 4) (Optional) Download pretrained checkpoint

```
will update later
```

## Train and Evaluate

### 1) Train
```
python train.py
```
Training the model will take about 80 hours on a single 4090 GPU.

### 2) Evaluate
For super-resolution using conditional diffusion, run:
```
python evaluate.py --method default --ckpt <path to pretrained model>
```

For super-resolution using conditional diffusion and measurement system, run:
```
python evaluate.py --method ddnm --ckpt <path to pretrained model>
```