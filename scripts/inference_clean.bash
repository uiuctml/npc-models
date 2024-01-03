#!/usr/bin/env bash

cd ../src

# 42    RBFT    RMFT    test
./inference_attribute.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -s 42

# 42    VBFT    VMFT    test
./inference_attribute.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -s 42

# 42    CRBFT   CRMFT   test
./inference_attribute.py -rb baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -rd decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -s 42

# 42    CVBFT   CVMFT   test
./inference_attribute.py -rb baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -rd decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -m ../../visat-spn/output/spn_matrices/matrix_a.pt -s 42
