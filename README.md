# deepGCFX
PyTorch implementation for our AAAI 2022 Paper "Graph-wise Common Latent Factor Extraction for Unsupervised Graph Representation Learning" [[PDF]](https://arxiv.org/pdf/2112.08830.pdf)

## Preparing the environment

1. Our implementation is in PyTorch running on GPUs. Use the provided [environment.yml](deepgcfx_environment.yml) to create a virtual environment using Anaconda.
2. Commands to automatically download graph-level and node-level datasets from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) are included in the ```main.py``` in each folder.

## Implementation Details

This repository contains implementations for all experiments we have used in our paper. [deepgcfx_graph](deepgcfx_graph) folder contains the implimentation for graph level tasks while [deepgcfx_node](deepgcfx_node) contains node-level implimentation.

Final hyper-parameter values to reproduce our results are provided in the Supplementary of our paper and will be updated in this repository soon.

## Training Steps

Go inside each respective folder and execute following commands.

#### Graph-level 
```python -u main.py --DS dataset_name --lr 0.001 --num-gc-layers 3 --hidden-dim 128 --batch_size 128 --num_epochs 500 --model_name deepgcfx_graph```

#### Node-level

```python -u main.py --lr 0.001 --num_epochs 2000 --num-gc-layers 1 --hidden-dim 512 --model_name deepgcfx_node```



