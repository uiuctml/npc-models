#!/usr/bin/env bash

cd ../src

# 42    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -s 42

# 42    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -s 42

# 42    CRBFT    test
./test_baseline.py -r baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -s 42

# 42    CVBFT    test
./test_baseline.py -r baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -s 42

# 84    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -s 84

# 84    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -s 84

# 84    CRBFT    test
./test_baseline.py -r baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler -s 84

# 84    CVBFT    test
./test_baseline.py -r baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton -s 84

# 126    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -s 126

# 126    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -s 126

# 126    CRBFT    test
./test_baseline.py -r baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler -s 126

# 126    CVBFT    test
./test_baseline.py -r baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton -s 126

# 168    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -s 168

# 168    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -s 168

# 168    CRBFT    test
./test_baseline.py -r baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler -s 168

# 168    CVBFT    test
./test_baseline.py -r baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton -s 168

# 210    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -s 210

# 210    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -s 210

# 210    CRBFT    test
./test_baseline.py -r baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler -s 210

# 210    CVBFT    test
./test_baseline.py -r baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton -s 210