# Transformer-Based Single Cell Integration

## Overview
This repository contains the implementation of our transformer-based model for single cell integration. Our approach leverages advanced transformer architectures to integrate single cell data from diverse sources, providing a unified view that is crucial for downstream analyses in genomics and biomedical research.

## Features
- **Transformer Model**: Utilizes the latest transformer models for effective data integration.
- **Data Compatibility**: Compatible with various single cell data types including scRNA-seq.
- **High Scalability**: Designed to handle large datasets efficiently.

## Installation
```bash
git clone https://github.com/C0nc/scInformer.git
cd scInformer
conda create -n leiden python=3.9
conda activate leiden
conda install cudatoolkit=11.7 -c pytorch -c nvidia
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --extra-index-url https://pypi.nvidia.com cudf-cu11==23.4.1 dask-cudf-cu11==23.4.1 cuml-cu11==23.4.1 cugraph-cu11==23.4.1 cucim==23.4.1
pip install einops ipdb pydance torchmetrics rapids-singlecell scvi-tools scib wandb hdf5plugin
pip install captum
```

## Usage

- **ig.ipynb** post-hoc IntegratedGradient analysis

- **batch.ipynb**  batch-wise token task

- **attention.ipynb**  attention visualization

- **run.py** train model

## Checkpoint 

https://drive.google.com/file/d/1XjRpMRMsqYyb-jcFvnB7yj_uwcAqHE0A/view?usp=sharing
