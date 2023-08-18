#!/usr/bin/env bash

cd ../src

ln -sfv test_42 ../../mapillary-dataset/images/split/corrupted/original/test

# 42    RMFT    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    VMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    RMFTL    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    VMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    CRMFT    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    CVMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    CRMFTL    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    CVMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

ln -sfv test_84 ../../mapillary-dataset/images/split/corrupted/original/test

# 84    RMFT    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    VMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    RMFTL    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    VMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    CRMFT    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    CVMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    CRMFTL    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

# 84    CVMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 84

ln -sfv test_126 ../../mapillary-dataset/images/split/corrupted/original/test

# 126    RMFT    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    VMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    RMFTL    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    VMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    CRMFT    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    CVMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    CRMFTL    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

# 126    CVMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 126

ln -sfv test_168 ../../mapillary-dataset/images/split/corrupted/original/test

# 168    RMFT    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    VMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    RMFTL    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    VMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    CRMFT    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    CVMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    CRMFTL    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

# 168    CVMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 168

ln -sfv test_210 ../../mapillary-dataset/images/split/corrupted/original/test

# 210    RMFT    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    VMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    RMFTL    Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    VMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    CRMFT    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    CVMFT    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    CRMFTL    Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210

# 210    CVMFTL    Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 210