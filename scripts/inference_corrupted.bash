#!/usr/bin/env bash

cd ../src

ln -sfv test_42 ../../mapillary-dataset/images/split/corrupted/original/test

# 42    RBFT    RMFT    Gaussian Noise 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 42

# 42    RBFT    RMFT    Shot Noise 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 42

# 42    RBFT    RMFT    Impulse Noise 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 42

# 42    RBFT    RMFT    Speckle Noise 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 42

# 42    RBFT    RMFT    Defocus Blur 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 42

# 42    RBFT    RMFT    Glass Blur 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 42

# 42    RBFT    RMFT    Motion Blur 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 42

# 42    RBFT    RMFT    Zoom Blur 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 42

# 42    RBFT    RMFT    Gaussian Blur 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 42

# 42    RBFT    RMFT    Snow 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 42

# 42    RBFT    RMFT    Frost 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 42

# 42    RBFT    RMFT    Fog 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 42

# 42    RBFT    RMFT    Spatter 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 42

# 42    RBFT    RMFT    Brightness 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 42

# 42    RBFT    RMFT    Contrast 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 42

# 42    RBFT    RMFT    Elastic Transform 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 42

# 42    RBFT    RMFT    Pixelate 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 42

# 42    RBFT    RMFT    JPEG 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_compression_5/ -s 42

# 42    RBFT    RMFT    Saturate 5
/inference.py -rb baseline.resnet152.2023.6.4.16.56.euler -rd decomposed.resnet152_mtl.2023.6.4.23.42.euler -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 42

# 42    VBFT    VMFT    Color
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/color/ -s 42

# 42    VBFT    VMFT    Gaussian Noise 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 42

# 42    VBFT    VMFT    Shot Noise 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 42

# 42    VBFT    VMFT    Impulse Noise 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 42

# 42    VBFT    VMFT    Speckle Noise 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 42

# 42    VBFT    VMFT    Defocus Blur 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 42

# 42    VBFT    VMFT    Glass Blur 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 42

# 42    VBFT    VMFT    Motion Blur 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 42

# 42    VBFT    VMFT    Zoom Blur 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 42

# 42    VBFT    VMFT    Gaussian Blur 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 42

# 42    VBFT    VMFT    Snow 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 42

# 42    VBFT    VMFT    Frost 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 42

# 42    VBFT    VMFT    Fog 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 42

# 42    VBFT    VMFT    Spatter 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 42

# 42    VBFT    VMFT    Brightness 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 42

# 42    VBFT    VMFT    Contrast 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 42

# 42    VBFT    VMFT    Elastic Transform 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 42

# 42    VBFT    VMFT    Pixelate 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 42

# 42    VBFT    VMFT    JPEG 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_compression_5/ -s 42

# 42    VBFT    VMFT    Saturate 5
./inference.py -rb baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -rd decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -m ../../visat-spn/output/spn_matrices/matrix_a.pt -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 42
