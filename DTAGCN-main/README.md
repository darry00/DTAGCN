# DTAGCN

The repo is the official implementation for the paper: **Dual-token Aware Graph Convolution Network with Interpretability for Long-term Multi-station
Irrigation Water Level Forecasting**. It currently includes code implementations for the following tasks:

> **Multivariate Forecasting**: We provide all scripts as well as datasets for the reproduction of forecasting results in this repo.

<p align="center">
<img src="./figures/algorithm.png" alt="" align=center />
</p>

## Usage 

Install Pytorch and other necessary dependencies.

```
pip install -r requirements.txt
```

Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Task: Multivariate forecasting with iTransformer
bash ./scripts/water/DTAGCN.sh
```
