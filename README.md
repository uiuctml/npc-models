# VISAT Benchmark Tools

## Overview

This codebase contains tools and scripts for performing the VISAT robustness benchmarks.

## Prerequisites

This codebase was developed on Ubuntu 20.04 LTS and requires the following packages:

 - (apt) python3-pip [20.0.2-5ubuntu1.8]
 - (apt) python3-opencv [4.2.0+dfsg-5]
 - (pip) scikit-image [0.18.0]
 - (pip) torch [4.64.1]
 - (pip) torchvision [4.64.1]
 - (pip) torchsummary [4.64.1]
 - (pip) tqdm [4.64.1]

Before attempting to launch a script, please refer to `header.py` and ensure that all relevant parameters, such as model type, hyperparameters, testing split path, etc., are properly set.

## Base Model Training

Under the project directory, the base models can be trained as follows:

```bash
cd src/
./train_baseline.py

```

## Multi-Task Learning (MTL) Model Training

Under the project directory, the MTL models can be trained as follows:

```bash
cd src/
./train_decomposed.py

```

## Base Model Testing

Under the project directory, the base models can be tested as follows:

```bash
cd src/
./test_baseline.py

```

## Multi-Task Learning (MTL) Model Testing

Under the project directory, the MTL models can be tested as follows:

```bash
cd src/
./test_decomposed.py

```
