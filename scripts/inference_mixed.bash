#!/usr/bin/env bash

cd ../src

# 42    RBFT    RMFT    Mixed
./inference.py -rb baseline.resnet152.ft.42.2024.1.3.15.16.newton -rd decomposed.resnet152_mtl.ft.42.2024.1.3.15.13.newton -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/mixed/ -s 42

# 42    VBFT    VMFT    Mixed
./inference.py -rb baseline.vit_b_32.ft.42.2024.1.3.23.46.newton -rd decomposed.vit_b_32_mtl.ft.42.2024.1.3.23.52.newton -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/mixed/ -s 42
