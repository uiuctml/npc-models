#!/usr/bin/env bash

cd ../src

# 42    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -s 42

# 42    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -s 42

# 42    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 42    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 84    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -s 84

# 84    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -s 84

# 84    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 84    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 126    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -s 126

# 126    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -s 126

# 126    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 126    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 168    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -s 168

# 168    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -s 168

# 168    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 168    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 210    RBFT    test
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -s 210

# 210    VBFT    test
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -s 210

# 210    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

# 210    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

