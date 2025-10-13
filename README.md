# Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (NeurIPS 2022)

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2022-blue.svg?style=flat-square)](#)
[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://arxiv.org/pdf/2205.13479)
[![arXiv](https://img.shields.io/badge/arXiv-2205.13479-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2205.13479)

This repository contains the code for the reproducibility of the experiments presented in the paper "Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations" (NeurIPS 2022). We propose a graph neural network that exploits a novel spatiotemporal attention to impute missing values leveraging only (sparse) valid observations.

**Authors**: [Ivan Marisca](mailto:ivan.marisca@usi.ch), [Andrea Cini](mailto:andrea.cini@usi.ch), Cesare Alippi

## SPIN in a nutshell

Spatiotemporal graphs are often highly sparse, with time series characterized by multiple, concurrent, and even long sequences of missing data, e.g., due to the unreliable underlying sensor network. In this context, autoregressive models can be brittle and exhibit unstable learning dynamics. The objective of this paper is, then, to tackle the problem of learning effective models to reconstruct, i.e., impute, missing data points by conditioning the reconstruction only on the available observations. In particular, we propose a novel class of attention-based architectures that, given a set of highly sparse discrete observations, learn a representation for points in time and space by exploiting a spatiotemporal diffusion architecture aligned with the imputation task. Representations are trained end-to-end to reconstruct observations w.r.t. the corresponding sensor and its neighboring nodes. Compared to the state of the art, our model handles sparse data without propagating prediction errors or requiring a bidirectional model to encode forward and backward time dependencies.

<div align=center>
	<img src="./sparse_att.png" alt="Example of the sparse spatiotemporal attention layer."/>
	<p align=left style="color: #777">Example of the sparse spatiotemporal attention layer. On the left, the input spatiotemporal graph, with time series associated with every node. On the right, how the layer acts to update target representation (highlighted by the green box), by simultaneously performing inter-node spatiotemporal cross-attention (red block) and intra-node temporal self-attention (violet block).</p>
</div>

---

## Directory structure

The directory is structured as follows:

```
.
├── config/
│   ├── imputation/
│   │   ├── brits.yaml
│   │   ├── grin.yaml
│   │   ├── saits.yaml
│   │   ├── spin.yaml
│   │   ├── spin_h.yaml
│   │   ├── spin_lane.yaml
│   │   └── transformer.yaml
│   └── inference.yaml
├── docs/
│   ├── data_structure.md
│   ├── lane_traffic_dataset.md
│   ├── node_connection_rules.md
│   └── user_defined_mask.md
├── examples/
│   ├── lane_interaction_example.py
│   ├── lane_traffic_example.py
│   ├── node_connection_example.py
│   ├── separated_data_example.py
│   └── user_defined_mask_example.py
├── experiments/
│   ├── run_imputation.py
│   └── run_inference.py
├── spin/
│   ├── baselines/
│   ├── datasets/
│   │   ├── lane_data_utils.py
│   │   └── lane_traffic_dataset.py
│   ├── imputers/
│   ├── layers/
│   ├── models/
│   └── ...
├── conda_env.yaml
└── tsl_config.yaml

```

## Installation

We provide a conda environment with all the project dependencies. To install the environment use:

```bash
conda env create -f conda_env.yml
conda activate spin
```

## Configuration files

The `config/` directory stores all the configuration files used to run the experiment. `config/imputation/` stores model configurations used for experiments on imputation.

## Python package `spin`

The support code, including models and baselines, are packed in a python package named `spin`.

## Experiments

The scripts used for the experiment in the paper are in the `experiments` folder.

* `run_imputation.py` is used to compute the metrics for the deep imputation methods. An example of usage is

	```bash
	conda activate spin
	python ./experiments/run_imputation.py --config imputation/spin.yaml --model-name spin --dataset-name bay_block
	```

* `run_inference.py` is used for the experiments on sparse datasets using pre-trained models. An example of usage is

	```bash
	conda activate spin
	python ./experiments/run_inference.py --config inference.yaml --model-name spin --dataset-name bay_point --exp-name {exp_name}
	```

## Lane-Level Traffic Dataset

This repository now includes support for lane-level traffic data with 10m×10s spatiotemporal grids. The dataset supports:

- **Separated data structure**: Static road data (lane_id, spatial_id, node_connections) and dynamic traffic data (timestamp, spatial_id, speed, spacing)
- **User-defined masks**: Precise control over which observations are known/unknown at each timestamp
- **Graph connectivity**: Node-level connections based on lane relationships

### Quick Start

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# Load dataset with user-defined mask
dataset = LaneTrafficDataset(
    static_data_path='data/static_road_data.csv',
    dynamic_data_path='data/dynamic_traffic_data.csv',
    mask_data_path='data/mask.csv',  # Optional: specify which observations are known
    time_col='timestamp',
    spatial_id_col='spatial_id',
    speed_col='speed',
    spacing_col='spacing'
)
```

### User-Defined Masks

The dataset supports custom observation masks in three formats:

1. **CSV Format**: Flexible control with `timestamp`, `spatial_id`, `is_observed` columns
2. **NPZ Format**: Direct mask matrix `[n_times, n_spaces]` or `[n_times, n_spaces, n_features]`
3. **PKL Format**: Python objects with mask arrays

See [`docs/user_defined_mask.md`](docs/user_defined_mask.md) for detailed documentation and [`examples/user_defined_mask_example.py`](examples/user_defined_mask_example.py) for examples.

### Documentation

- [`docs/data_structure.md`](docs/data_structure.md): Data structure overview
- [`docs/lane_traffic_dataset.md`](docs/lane_traffic_dataset.md): Dataset usage guide
- [`docs/node_connection_rules.md`](docs/node_connection_rules.md): Graph connectivity rules
- [`docs/user_defined_mask.md`](docs/user_defined_mask.md): User-defined mask guide

## Bibtex reference

If you find this code useful please consider to cite our paper:

```
@article{marisca2022learning,
  title={Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations},
  author={Marisca, Ivan and Cini, Andrea and Alippi, Cesare},
  journal={arXiv preprint arXiv:2205.13479},
  year={2022}
}
```
 