#!/usr/bin/env bash

cd ../src

# 42    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 42    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 42    RBFT    CRBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler.best.tar.pt -s 42 -a 1

# 42    RBFT    CVBFT
./test_baseline.py -r baseline.resnet152.2023.6.4.16.56.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton.best.tar.pt -s 42 -a 1

# 42    VBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler.best.tar.pt -s 42 -a 1

# 42    VBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton.best.tar.pt -s 42 -a 1

# 42    CRBFT    RBFT
./test_baseline.py -r baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    CRBFT    VBFT
./test_baseline.py -r baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 42    CVBFT    RBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.4.16.56.euler.best.tar.pt -s 42 -a 1

# 42    CVBFT    VBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2.best.tar.pt -s 42 -a 1

# 42    CRBFT    CRBFT
./test_baseline.py -r baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler.best.tar.pt -s 42 -a 1

# 42    CRBFT    CVBFT
./test_baseline.py -r baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton.best.tar.pt -s 42 -a 1

# 42    CVBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.42.2023.8.12.14.29.euler.best.tar.pt -s 42 -a 1

# 42    CVBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.42.2023.8.12.19.55.newton.best.tar.pt -s 42 -a 1

# 84    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 84    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 84    RBFT    CRBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler.best.tar.pt -s 84 -a 1

# 84    RBFT    CVBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.15.29.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton.best.tar.pt -s 84 -a 1

# 84    VBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler.best.tar.pt -s 84 -a 1

# 84    VBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton.best.tar.pt -s 84 -a 1

# 84    CRBFT    RBFT
./test_baseline.py -r baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    CRBFT    VBFT
./test_baseline.py -r baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 84    CVBFT    RBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.15.29.euler.best.tar.pt -s 84 -a 1

# 84    CVBFT    VBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2.best.tar.pt -s 84 -a 1

# 84    CRBFT    CRBFT
./test_baseline.py -r baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler.best.tar.pt -s 84 -a 1

# 84    CRBFT    CVBFT
./test_baseline.py -r baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton.best.tar.pt -s 84 -a 1

# 84    CVBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.84.2023.8.13.7.32.euler.best.tar.pt -s 84 -a 1

# 84    CVBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.84.2023.8.13.1.32.newton.best.tar.pt -s 84 -a 1

# 126    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 126    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 126    RBFT    CRBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler.best.tar.pt -s 126 -a 1

# 126    RBFT    CVBFT
./test_baseline.py -r baseline.resnet152.2023.6.7.22.53.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton.best.tar.pt -s 126 -a 1

# 126    VBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler.best.tar.pt -s 126 -a 1

# 126    VBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton.best.tar.pt -s 126 -a 1

# 126    CRBFT    RBFT
./test_baseline.py -r baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    CRBFT    VBFT
./test_baseline.py -r baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 126    CVBFT    RBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.7.22.53.euler.best.tar.pt -s 126 -a 1

# 126    CVBFT    VBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2.best.tar.pt -s 126 -a 1

# 126    CRBFT    CRBFT
./test_baseline.py -r baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler.best.tar.pt -s 126 -a 1

# 126    CRBFT    CVBFT
./test_baseline.py -r baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton.best.tar.pt -s 126 -a 1

# 126    CVBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.126.2023.8.13.11.46.euler.best.tar.pt -s 126 -a 1

# 126    CVBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.126.2023.8.13.3.22.newton.best.tar.pt -s 126 -a 1

# 168    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 168    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 168    RBFT    CRBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler.best.tar.pt -s 168 -a 1

# 168    RBFT    CVBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.8.49.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton.best.tar.pt -s 168 -a 1

# 168    VBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler.best.tar.pt -s 168 -a 1

# 168    VBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton.best.tar.pt -s 168 -a 1

# 168    CRBFT    RBFT
./test_baseline.py -r baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    CRBFT    VBFT
./test_baseline.py -r baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 168    CVBFT    RBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.8.49.euler.best.tar.pt -s 168 -a 1

# 168    CVBFT    VBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2.best.tar.pt -s 168 -a 1

# 168    CRBFT    CRBFT
./test_baseline.py -r baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler.best.tar.pt -s 168 -a 1

# 168    CRBFT    CVBFT
./test_baseline.py -r baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton.best.tar.pt -s 168 -a 1

# 168    CVBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.168.2023.8.13.15.58.euler.best.tar.pt -s 168 -a 1

# 168    CVBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.168.2023.8.13.5.12.newton.best.tar.pt -s 168 -a 1

# 210    RBFT    RBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    RBFT    VBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

# 210    VBFT    RBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    VBFT    VBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

# 210    RBFT    CRBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler.best.tar.pt -s 210 -a 1

# 210    RBFT    CVBFT
./test_baseline.py -r baseline.resnet152.2023.6.8.18.33.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton.best.tar.pt -s 210 -a 1

# 210    VBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler.best.tar.pt -s 210 -a 1

# 210    VBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton.best.tar.pt -s 210 -a 1

# 210    CRBFT    RBFT
./test_baseline.py -r baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    CRBFT    VBFT
./test_baseline.py -r baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

# 210    CVBFT    RBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet152.2023.6.8.18.33.euler.best.tar.pt -s 210 -a 1

# 210    CVBFT    VBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2.best.tar.pt -s 210 -a 1

# 210    CRBFT    CRBFT
./test_baseline.py -r baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler.best.tar.pt -s 210 -a 1

# 210    CRBFT    CVBFT
./test_baseline.py -r baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton.best.tar.pt -s 210 -a 1

# 210    CVBFT    CRBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.resnet101_clip.ft.210.2023.8.13.20.14.euler.best.tar.pt -s 210 -a 1

# 210    CVBFT    CVBFT
./test_baseline.py -r baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton -i ../../mapillary-dataset/images/split/adversarial/pgd/original/baseline.vit_b_32_clip.ft.210.2023.8.13.7.2.newton.best.tar.pt -s 210 -a 1