#!/usr/bin/env bash

cd ../src

# 42    RBFT    test
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -s 42

# 42    VBFT    test
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -s 42

# 42    RBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 42

# 42    RBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 42

# 42    RBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 42

# 42    RBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 42

# 42    RBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 42

# 42    RBFT    Shot Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 42

# 42    RBFT    Shot Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 42

# 42    RBFT    Shot Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 42

# 42    RBFT    Shot Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 42

# 42    RBFT    Shot Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 42

# 42    RBFT    Impulse Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 42

# 42    RBFT    Impulse Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 42

# 42    RBFT    Impulse Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 42

# 42    RBFT    Impulse Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 42

# 42    RBFT    Impulse Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 42

# 42    RBFT    Speckle Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 42

# 42    RBFT    Speckle Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 42

# 42    RBFT    Speckle Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 42

# 42    RBFT    Speckle Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 42

# 42    RBFT    Speckle Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 42

# 42    RBFT    Defocus Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 42

# 42    RBFT    Defocus Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 42

# 42    RBFT    Defocus Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 42

# 42    RBFT    Defocus Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 42

# 42    RBFT    Defocus Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 42

# 42    RBFT    Glass Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 42

# 42    RBFT    Glass Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 42

# 42    RBFT    Glass Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 42

# 42    RBFT    Glass Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 42

# 42    RBFT    Glass Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 42

# 42    RBFT    Motion Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 42

# 42    RBFT    Motion Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 42

# 42    RBFT    Motion Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 42

# 42    RBFT    Motion Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 42

# 42    RBFT    Motion Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 42

# 42    RBFT    Zoom Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 42

# 42    RBFT    Zoom Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 42

# 42    RBFT    Zoom Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 42

# 42    RBFT    Zoom Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 42

# 42    RBFT    Zoom Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 42

# 42    RBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 42

# 42    RBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 42

# 42    RBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 42

# 42    RBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 42

# 42    RBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 42

# 42    RBFT    Snow 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 42

# 42    RBFT    Snow 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 42

# 42    RBFT    Snow 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 42

# 42    RBFT    Snow 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 42

# 42    RBFT    Snow 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 42

# 42    RBFT    Frost 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 42

# 42    RBFT    Frost 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 42

# 42    RBFT    Frost 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 42

# 42    RBFT    Frost 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 42

# 42    RBFT    Frost 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 42

# 42    RBFT    Fog 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 42

# 42    RBFT    Fog 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 42

# 42    RBFT    Fog 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 42

# 42    RBFT    Fog 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 42

# 42    RBFT    Fog 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 42

# 42    RBFT    Spatter 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 42

# 42    RBFT    Spatter 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 42

# 42    RBFT    Spatter 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 42

# 42    RBFT    Spatter 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 42

# 42    RBFT    Spatter 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 42

# 42    RBFT    Brightness 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 42

# 42    RBFT    Brightness 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 42

# 42    RBFT    Brightness 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 42

# 42    RBFT    Brightness 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 42

# 42    RBFT    Brightness 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 42

# 42    RBFT    Contrast 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 42

# 42    RBFT    Contrast 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 42

# 42    RBFT    Contrast 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 42

# 42    RBFT    Contrast 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 42

# 42    RBFT    Contrast 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 42

# 42    RBFT    Elastic Transform 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 42

# 42    RBFT    Elastic Transform 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 42

# 42    RBFT    Elastic Transform 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 42

# 42    RBFT    Elastic Transform 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 42

# 42    RBFT    Elastic Transform 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 42

# 42    RBFT    Pixelate 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 42

# 42    RBFT    Pixelate 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 42

# 42    RBFT    Pixelate 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 42

# 42    RBFT    Pixelate 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 42

# 42    RBFT    Pixelate 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 42

# 42    RBFT    JPEG 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 42

# 42    RBFT    JPEG 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 42

# 42    RBFT    JPEG 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 42

# 42    RBFT    JPEG 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 42

# 42    RBFT    JPEG 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 42

# 42    RBFT    Saturate 1
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 42

# 42    RBFT    Saturate 2
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 42

# 42    RBFT    Saturate 3
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 42

# 42    RBFT    Saturate 4
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 42

# 42    RBFT    Saturate 5
./test_decomposed.py -r baseline.resnet152.2023.6.4.16.56.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 42

# 42    VBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 42

# 42    VBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 42

# 42    VBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 42

# 42    VBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 42

# 42    VBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 42

# 42    VBFT    Shot Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 42

# 42    VBFT    Shot Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 42

# 42    VBFT    Shot Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 42

# 42    VBFT    Shot Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 42

# 42    VBFT    Shot Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 42

# 42    VBFT    Impulse Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 42

# 42    VBFT    Impulse Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 42

# 42    VBFT    Impulse Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 42

# 42    VBFT    Impulse Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 42

# 42    VBFT    Impulse Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 42

# 42    VBFT    Speckle Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 42

# 42    VBFT    Speckle Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 42

# 42    VBFT    Speckle Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 42

# 42    VBFT    Speckle Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 42

# 42    VBFT    Speckle Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 42

# 42    VBFT    Defocus Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 42

# 42    VBFT    Defocus Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 42

# 42    VBFT    Defocus Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 42

# 42    VBFT    Defocus Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 42

# 42    VBFT    Defocus Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 42

# 42    VBFT    Glass Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 42

# 42    VBFT    Glass Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 42

# 42    VBFT    Glass Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 42

# 42    VBFT    Glass Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 42

# 42    VBFT    Glass Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 42

# 42    VBFT    Motion Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 42

# 42    VBFT    Motion Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 42

# 42    VBFT    Motion Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 42

# 42    VBFT    Motion Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 42

# 42    VBFT    Motion Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 42

# 42    VBFT    Zoom Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 42

# 42    VBFT    Zoom Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 42

# 42    VBFT    Zoom Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 42

# 42    VBFT    Zoom Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 42

# 42    VBFT    Zoom Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 42

# 42    VBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 42

# 42    VBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 42

# 42    VBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 42

# 42    VBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 42

# 42    VBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 42

# 42    VBFT    Snow 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 42

# 42    VBFT    Snow 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 42

# 42    VBFT    Snow 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 42

# 42    VBFT    Snow 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 42

# 42    VBFT    Snow 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 42

# 42    VBFT    Frost 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 42

# 42    VBFT    Frost 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 42

# 42    VBFT    Frost 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 42

# 42    VBFT    Frost 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 42

# 42    VBFT    Frost 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 42

# 42    VBFT    Fog 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 42

# 42    VBFT    Fog 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 42

# 42    VBFT    Fog 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 42

# 42    VBFT    Fog 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 42

# 42    VBFT    Fog 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 42

# 42    VBFT    Spatter 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 42

# 42    VBFT    Spatter 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 42

# 42    VBFT    Spatter 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 42

# 42    VBFT    Spatter 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 42

# 42    VBFT    Spatter 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 42

# 42    VBFT    Brightness 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 42

# 42    VBFT    Brightness 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 42

# 42    VBFT    Brightness 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 42

# 42    VBFT    Brightness 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 42

# 42    VBFT    Brightness 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 42

# 42    VBFT    Contrast 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 42

# 42    VBFT    Contrast 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 42

# 42    VBFT    Contrast 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 42

# 42    VBFT    Contrast 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 42

# 42    VBFT    Contrast 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 42

# 42    VBFT    Elastic Transform 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 42

# 42    VBFT    Elastic Transform 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 42

# 42    VBFT    Elastic Transform 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 42

# 42    VBFT    Elastic Transform 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 42

# 42    VBFT    Elastic Transform 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 42

# 42    VBFT    Pixelate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 42

# 42    VBFT    Pixelate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 42

# 42    VBFT    Pixelate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 42

# 42    VBFT    Pixelate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 42

# 42    VBFT    Pixelate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 42

# 42    VBFT    JPEG 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 42

# 42    VBFT    JPEG 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 42

# 42    VBFT    JPEG 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 42

# 42    VBFT    JPEG 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 42

# 42    VBFT    JPEG 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 42

# 42    VBFT    Saturate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 42

# 42    VBFT    Saturate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 42

# 42    VBFT    Saturate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 42

# 42    VBFT    Saturate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 42

# 42    VBFT    Saturate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.4.16.27.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 42

# 84    RBFT    test
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -s 84

# 84    VBFT    test
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -s 84

# 84    RBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 84

# 84    RBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 84

# 84    RBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 84

# 84    RBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 84

# 84    RBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 84

# 84    RBFT    Shot Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 84

# 84    RBFT    Shot Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 84

# 84    RBFT    Shot Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 84

# 84    RBFT    Shot Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 84

# 84    RBFT    Shot Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 84

# 84    RBFT    Impulse Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 84

# 84    RBFT    Impulse Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 84

# 84    RBFT    Impulse Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 84

# 84    RBFT    Impulse Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 84

# 84    RBFT    Impulse Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 84

# 84    RBFT    Speckle Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 84

# 84    RBFT    Speckle Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 84

# 84    RBFT    Speckle Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 84

# 84    RBFT    Speckle Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 84

# 84    RBFT    Speckle Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 84

# 84    RBFT    Defocus Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 84

# 84    RBFT    Defocus Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 84

# 84    RBFT    Defocus Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 84

# 84    RBFT    Defocus Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 84

# 84    RBFT    Defocus Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 84

# 84    RBFT    Glass Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 84

# 84    RBFT    Glass Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 84

# 84    RBFT    Glass Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 84

# 84    RBFT    Glass Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 84

# 84    RBFT    Glass Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 84

# 84    RBFT    Motion Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 84

# 84    RBFT    Motion Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 84

# 84    RBFT    Motion Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 84

# 84    RBFT    Motion Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 84

# 84    RBFT    Motion Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 84

# 84    RBFT    Zoom Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 84

# 84    RBFT    Zoom Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 84

# 84    RBFT    Zoom Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 84

# 84    RBFT    Zoom Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 84

# 84    RBFT    Zoom Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 84

# 84    RBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 84

# 84    RBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 84

# 84    RBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 84

# 84    RBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 84

# 84    RBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 84

# 84    RBFT    Snow 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 84

# 84    RBFT    Snow 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 84

# 84    RBFT    Snow 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 84

# 84    RBFT    Snow 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 84

# 84    RBFT    Snow 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 84

# 84    RBFT    Frost 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 84

# 84    RBFT    Frost 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 84

# 84    RBFT    Frost 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 84

# 84    RBFT    Frost 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 84

# 84    RBFT    Frost 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 84

# 84    RBFT    Fog 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 84

# 84    RBFT    Fog 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 84

# 84    RBFT    Fog 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 84

# 84    RBFT    Fog 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 84

# 84    RBFT    Fog 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 84

# 84    RBFT    Spatter 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 84

# 84    RBFT    Spatter 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 84

# 84    RBFT    Spatter 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 84

# 84    RBFT    Spatter 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 84

# 84    RBFT    Spatter 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 84

# 84    RBFT    Brightness 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 84

# 84    RBFT    Brightness 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 84

# 84    RBFT    Brightness 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 84

# 84    RBFT    Brightness 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 84

# 84    RBFT    Brightness 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 84

# 84    RBFT    Contrast 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 84

# 84    RBFT    Contrast 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 84

# 84    RBFT    Contrast 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 84

# 84    RBFT    Contrast 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 84

# 84    RBFT    Contrast 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 84

# 84    RBFT    Elastic Transform 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 84

# 84    RBFT    Elastic Transform 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 84

# 84    RBFT    Elastic Transform 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 84

# 84    RBFT    Elastic Transform 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 84

# 84    RBFT    Elastic Transform 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 84

# 84    RBFT    Pixelate 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 84

# 84    RBFT    Pixelate 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 84

# 84    RBFT    Pixelate 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 84

# 84    RBFT    Pixelate 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 84

# 84    RBFT    Pixelate 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 84

# 84    RBFT    JPEG 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 84

# 84    RBFT    JPEG 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 84

# 84    RBFT    JPEG 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 84

# 84    RBFT    JPEG 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 84

# 84    RBFT    JPEG 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 84

# 84    RBFT    Saturate 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 84

# 84    RBFT    Saturate 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 84

# 84    RBFT    Saturate 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 84

# 84    RBFT    Saturate 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 84

# 84    RBFT    Saturate 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.15.29.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 84

# 84    VBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 84

# 84    VBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 84

# 84    VBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 84

# 84    VBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 84

# 84    VBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 84

# 84    VBFT    Shot Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 84

# 84    VBFT    Shot Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 84

# 84    VBFT    Shot Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 84

# 84    VBFT    Shot Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 84

# 84    VBFT    Shot Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 84

# 84    VBFT    Impulse Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 84

# 84    VBFT    Impulse Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 84

# 84    VBFT    Impulse Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 84

# 84    VBFT    Impulse Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 84

# 84    VBFT    Impulse Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 84

# 84    VBFT    Speckle Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 84

# 84    VBFT    Speckle Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 84

# 84    VBFT    Speckle Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 84

# 84    VBFT    Speckle Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 84

# 84    VBFT    Speckle Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 84

# 84    VBFT    Defocus Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 84

# 84    VBFT    Defocus Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 84

# 84    VBFT    Defocus Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 84

# 84    VBFT    Defocus Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 84

# 84    VBFT    Defocus Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 84

# 84    VBFT    Glass Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 84

# 84    VBFT    Glass Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 84

# 84    VBFT    Glass Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 84

# 84    VBFT    Glass Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 84

# 84    VBFT    Glass Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 84

# 84    VBFT    Motion Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 84

# 84    VBFT    Motion Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 84

# 84    VBFT    Motion Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 84

# 84    VBFT    Motion Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 84

# 84    VBFT    Motion Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 84

# 84    VBFT    Zoom Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 84

# 84    VBFT    Zoom Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 84

# 84    VBFT    Zoom Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 84

# 84    VBFT    Zoom Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 84

# 84    VBFT    Zoom Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 84

# 84    VBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 84

# 84    VBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 84

# 84    VBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 84

# 84    VBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 84

# 84    VBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 84

# 84    VBFT    Snow 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 84

# 84    VBFT    Snow 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 84

# 84    VBFT    Snow 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 84

# 84    VBFT    Snow 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 84

# 84    VBFT    Snow 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 84

# 84    VBFT    Frost 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 84

# 84    VBFT    Frost 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 84

# 84    VBFT    Frost 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 84

# 84    VBFT    Frost 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 84

# 84    VBFT    Frost 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 84

# 84    VBFT    Fog 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 84

# 84    VBFT    Fog 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 84

# 84    VBFT    Fog 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 84

# 84    VBFT    Fog 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 84

# 84    VBFT    Fog 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 84

# 84    VBFT    Spatter 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 84

# 84    VBFT    Spatter 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 84

# 84    VBFT    Spatter 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 84

# 84    VBFT    Spatter 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 84

# 84    VBFT    Spatter 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 84

# 84    VBFT    Brightness 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 84

# 84    VBFT    Brightness 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 84

# 84    VBFT    Brightness 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 84

# 84    VBFT    Brightness 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 84

# 84    VBFT    Brightness 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 84

# 84    VBFT    Contrast 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 84

# 84    VBFT    Contrast 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 84

# 84    VBFT    Contrast 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 84

# 84    VBFT    Contrast 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 84

# 84    VBFT    Contrast 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 84

# 84    VBFT    Elastic Transform 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 84

# 84    VBFT    Elastic Transform 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 84

# 84    VBFT    Elastic Transform 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 84

# 84    VBFT    Elastic Transform 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 84

# 84    VBFT    Elastic Transform 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 84

# 84    VBFT    Pixelate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 84

# 84    VBFT    Pixelate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 84

# 84    VBFT    Pixelate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 84

# 84    VBFT    Pixelate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 84

# 84    VBFT    Pixelate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 84

# 84    VBFT    JPEG 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 84

# 84    VBFT    JPEG 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 84

# 84    VBFT    JPEG 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 84

# 84    VBFT    JPEG 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 84

# 84    VBFT    JPEG 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 84

# 84    VBFT    Saturate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 84

# 84    VBFT    Saturate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 84

# 84    VBFT    Saturate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 84

# 84    VBFT    Saturate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 84

# 84    VBFT    Saturate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.15.25.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 84

# 126    RBFT    test
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -s 126

# 126    VBFT    test
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -s 126

# 126    RBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 126

# 126    RBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 126

# 126    RBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 126

# 126    RBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 126

# 126    RBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 126

# 126    RBFT    Shot Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 126

# 126    RBFT    Shot Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 126

# 126    RBFT    Shot Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 126

# 126    RBFT    Shot Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 126

# 126    RBFT    Shot Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 126

# 126    RBFT    Impulse Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 126

# 126    RBFT    Impulse Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 126

# 126    RBFT    Impulse Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 126

# 126    RBFT    Impulse Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 126

# 126    RBFT    Impulse Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 126

# 126    RBFT    Speckle Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 126

# 126    RBFT    Speckle Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 126

# 126    RBFT    Speckle Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 126

# 126    RBFT    Speckle Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 126

# 126    RBFT    Speckle Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 126

# 126    RBFT    Defocus Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 126

# 126    RBFT    Defocus Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 126

# 126    RBFT    Defocus Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 126

# 126    RBFT    Defocus Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 126

# 126    RBFT    Defocus Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 126

# 126    RBFT    Glass Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 126

# 126    RBFT    Glass Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 126

# 126    RBFT    Glass Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 126

# 126    RBFT    Glass Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 126

# 126    RBFT    Glass Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 126

# 126    RBFT    Motion Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 126

# 126    RBFT    Motion Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 126

# 126    RBFT    Motion Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 126

# 126    RBFT    Motion Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 126

# 126    RBFT    Motion Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 126

# 126    RBFT    Zoom Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 126

# 126    RBFT    Zoom Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 126

# 126    RBFT    Zoom Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 126

# 126    RBFT    Zoom Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 126

# 126    RBFT    Zoom Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 126

# 126    RBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 126

# 126    RBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 126

# 126    RBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 126

# 126    RBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 126

# 126    RBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 126

# 126    RBFT    Snow 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 126

# 126    RBFT    Snow 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 126

# 126    RBFT    Snow 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 126

# 126    RBFT    Snow 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 126

# 126    RBFT    Snow 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 126

# 126    RBFT    Frost 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 126

# 126    RBFT    Frost 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 126

# 126    RBFT    Frost 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 126

# 126    RBFT    Frost 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 126

# 126    RBFT    Frost 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 126

# 126    RBFT    Fog 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 126

# 126    RBFT    Fog 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 126

# 126    RBFT    Fog 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 126

# 126    RBFT    Fog 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 126

# 126    RBFT    Fog 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 126

# 126    RBFT    Spatter 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 126

# 126    RBFT    Spatter 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 126

# 126    RBFT    Spatter 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 126

# 126    RBFT    Spatter 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 126

# 126    RBFT    Spatter 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 126

# 126    RBFT    Brightness 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 126

# 126    RBFT    Brightness 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 126

# 126    RBFT    Brightness 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 126

# 126    RBFT    Brightness 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 126

# 126    RBFT    Brightness 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 126

# 126    RBFT    Contrast 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 126

# 126    RBFT    Contrast 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 126

# 126    RBFT    Contrast 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 126

# 126    RBFT    Contrast 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 126

# 126    RBFT    Contrast 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 126

# 126    RBFT    Elastic Transform 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 126

# 126    RBFT    Elastic Transform 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 126

# 126    RBFT    Elastic Transform 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 126

# 126    RBFT    Elastic Transform 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 126

# 126    RBFT    Elastic Transform 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 126

# 126    RBFT    Pixelate 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 126

# 126    RBFT    Pixelate 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 126

# 126    RBFT    Pixelate 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 126

# 126    RBFT    Pixelate 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 126

# 126    RBFT    Pixelate 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 126

# 126    RBFT    JPEG 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 126

# 126    RBFT    JPEG 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 126

# 126    RBFT    JPEG 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 126

# 126    RBFT    JPEG 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 126

# 126    RBFT    JPEG 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 126

# 126    RBFT    Saturate 1
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 126

# 126    RBFT    Saturate 2
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 126

# 126    RBFT    Saturate 3
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 126

# 126    RBFT    Saturate 4
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 126

# 126    RBFT    Saturate 5
./test_decomposed.py -r baseline.resnet152.2023.6.7.22.53.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 126

# 126    VBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 126

# 126    VBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 126

# 126    VBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 126

# 126    VBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 126

# 126    VBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 126

# 126    VBFT    Shot Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 126

# 126    VBFT    Shot Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 126

# 126    VBFT    Shot Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 126

# 126    VBFT    Shot Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 126

# 126    VBFT    Shot Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 126

# 126    VBFT    Impulse Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 126

# 126    VBFT    Impulse Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 126

# 126    VBFT    Impulse Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 126

# 126    VBFT    Impulse Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 126

# 126    VBFT    Impulse Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 126

# 126    VBFT    Speckle Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 126

# 126    VBFT    Speckle Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 126

# 126    VBFT    Speckle Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 126

# 126    VBFT    Speckle Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 126

# 126    VBFT    Speckle Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 126

# 126    VBFT    Defocus Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 126

# 126    VBFT    Defocus Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 126

# 126    VBFT    Defocus Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 126

# 126    VBFT    Defocus Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 126

# 126    VBFT    Defocus Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 126

# 126    VBFT    Glass Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 126

# 126    VBFT    Glass Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 126

# 126    VBFT    Glass Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 126

# 126    VBFT    Glass Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 126

# 126    VBFT    Glass Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 126

# 126    VBFT    Motion Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 126

# 126    VBFT    Motion Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 126

# 126    VBFT    Motion Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 126

# 126    VBFT    Motion Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 126

# 126    VBFT    Motion Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 126

# 126    VBFT    Zoom Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 126

# 126    VBFT    Zoom Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 126

# 126    VBFT    Zoom Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 126

# 126    VBFT    Zoom Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 126

# 126    VBFT    Zoom Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 126

# 126    VBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 126

# 126    VBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 126

# 126    VBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 126

# 126    VBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 126

# 126    VBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 126

# 126    VBFT    Snow 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 126

# 126    VBFT    Snow 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 126

# 126    VBFT    Snow 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 126

# 126    VBFT    Snow 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 126

# 126    VBFT    Snow 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 126

# 126    VBFT    Frost 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 126

# 126    VBFT    Frost 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 126

# 126    VBFT    Frost 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 126

# 126    VBFT    Frost 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 126

# 126    VBFT    Frost 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 126

# 126    VBFT    Fog 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 126

# 126    VBFT    Fog 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 126

# 126    VBFT    Fog 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 126

# 126    VBFT    Fog 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 126

# 126    VBFT    Fog 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 126

# 126    VBFT    Spatter 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 126

# 126    VBFT    Spatter 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 126

# 126    VBFT    Spatter 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 126

# 126    VBFT    Spatter 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 126

# 126    VBFT    Spatter 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 126

# 126    VBFT    Brightness 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 126

# 126    VBFT    Brightness 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 126

# 126    VBFT    Brightness 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 126

# 126    VBFT    Brightness 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 126

# 126    VBFT    Brightness 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 126

# 126    VBFT    Contrast 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 126

# 126    VBFT    Contrast 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 126

# 126    VBFT    Contrast 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 126

# 126    VBFT    Contrast 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 126

# 126    VBFT    Contrast 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 126

# 126    VBFT    Elastic Transform 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 126

# 126    VBFT    Elastic Transform 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 126

# 126    VBFT    Elastic Transform 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 126

# 126    VBFT    Elastic Transform 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 126

# 126    VBFT    Elastic Transform 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 126

# 126    VBFT    Pixelate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 126

# 126    VBFT    Pixelate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 126

# 126    VBFT    Pixelate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 126

# 126    VBFT    Pixelate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 126

# 126    VBFT    Pixelate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 126

# 126    VBFT    JPEG 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 126

# 126    VBFT    JPEG 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 126

# 126    VBFT    JPEG 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 126

# 126    VBFT    JPEG 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 126

# 126    VBFT    JPEG 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 126

# 126    VBFT    Saturate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 126

# 126    VBFT    Saturate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 126

# 126    VBFT    Saturate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 126

# 126    VBFT    Saturate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 126

# 126    VBFT    Saturate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.7.21.58.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 126

# 168    RBFT    test
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -s 168

# 168    VBFT    test
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -s 168

# 168    RBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 168

# 168    RBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 168

# 168    RBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 168

# 168    RBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 168

# 168    RBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 168

# 168    RBFT    Shot Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 168

# 168    RBFT    Shot Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 168

# 168    RBFT    Shot Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 168

# 168    RBFT    Shot Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 168

# 168    RBFT    Shot Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 168

# 168    RBFT    Impulse Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 168

# 168    RBFT    Impulse Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 168

# 168    RBFT    Impulse Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 168

# 168    RBFT    Impulse Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 168

# 168    RBFT    Impulse Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 168

# 168    RBFT    Speckle Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 168

# 168    RBFT    Speckle Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 168

# 168    RBFT    Speckle Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 168

# 168    RBFT    Speckle Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 168

# 168    RBFT    Speckle Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 168

# 168    RBFT    Defocus Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 168

# 168    RBFT    Defocus Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 168

# 168    RBFT    Defocus Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 168

# 168    RBFT    Defocus Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 168

# 168    RBFT    Defocus Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 168

# 168    RBFT    Glass Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 168

# 168    RBFT    Glass Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 168

# 168    RBFT    Glass Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 168

# 168    RBFT    Glass Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 168

# 168    RBFT    Glass Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 168

# 168    RBFT    Motion Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 168

# 168    RBFT    Motion Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 168

# 168    RBFT    Motion Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 168

# 168    RBFT    Motion Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 168

# 168    RBFT    Motion Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 168

# 168    RBFT    Zoom Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 168

# 168    RBFT    Zoom Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 168

# 168    RBFT    Zoom Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 168

# 168    RBFT    Zoom Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 168

# 168    RBFT    Zoom Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 168

# 168    RBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 168

# 168    RBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 168

# 168    RBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 168

# 168    RBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 168

# 168    RBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 168

# 168    RBFT    Snow 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 168

# 168    RBFT    Snow 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 168

# 168    RBFT    Snow 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 168

# 168    RBFT    Snow 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 168

# 168    RBFT    Snow 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 168

# 168    RBFT    Frost 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 168

# 168    RBFT    Frost 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 168

# 168    RBFT    Frost 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 168

# 168    RBFT    Frost 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 168

# 168    RBFT    Frost 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 168

# 168    RBFT    Fog 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 168

# 168    RBFT    Fog 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 168

# 168    RBFT    Fog 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 168

# 168    RBFT    Fog 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 168

# 168    RBFT    Fog 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 168

# 168    RBFT    Spatter 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 168

# 168    RBFT    Spatter 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 168

# 168    RBFT    Spatter 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 168

# 168    RBFT    Spatter 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 168

# 168    RBFT    Spatter 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 168

# 168    RBFT    Brightness 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 168

# 168    RBFT    Brightness 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 168

# 168    RBFT    Brightness 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 168

# 168    RBFT    Brightness 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 168

# 168    RBFT    Brightness 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 168

# 168    RBFT    Contrast 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 168

# 168    RBFT    Contrast 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 168

# 168    RBFT    Contrast 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 168

# 168    RBFT    Contrast 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 168

# 168    RBFT    Contrast 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 168

# 168    RBFT    Elastic Transform 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 168

# 168    RBFT    Elastic Transform 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 168

# 168    RBFT    Elastic Transform 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 168

# 168    RBFT    Elastic Transform 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 168

# 168    RBFT    Elastic Transform 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 168

# 168    RBFT    Pixelate 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 168

# 168    RBFT    Pixelate 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 168

# 168    RBFT    Pixelate 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 168

# 168    RBFT    Pixelate 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 168

# 168    RBFT    Pixelate 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 168

# 168    RBFT    JPEG 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 168

# 168    RBFT    JPEG 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 168

# 168    RBFT    JPEG 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 168

# 168    RBFT    JPEG 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 168

# 168    RBFT    JPEG 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 168

# 168    RBFT    Saturate 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 168

# 168    RBFT    Saturate 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 168

# 168    RBFT    Saturate 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 168

# 168    RBFT    Saturate 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 168

# 168    RBFT    Saturate 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.8.49.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 168

# 168    VBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 168

# 168    VBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 168

# 168    VBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 168

# 168    VBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 168

# 168    VBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 168

# 168    VBFT    Shot Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 168

# 168    VBFT    Shot Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 168

# 168    VBFT    Shot Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 168

# 168    VBFT    Shot Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 168

# 168    VBFT    Shot Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 168

# 168    VBFT    Impulse Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 168

# 168    VBFT    Impulse Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 168

# 168    VBFT    Impulse Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 168

# 168    VBFT    Impulse Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 168

# 168    VBFT    Impulse Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 168

# 168    VBFT    Speckle Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 168

# 168    VBFT    Speckle Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 168

# 168    VBFT    Speckle Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 168

# 168    VBFT    Speckle Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 168

# 168    VBFT    Speckle Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 168

# 168    VBFT    Defocus Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 168

# 168    VBFT    Defocus Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 168

# 168    VBFT    Defocus Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 168

# 168    VBFT    Defocus Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 168

# 168    VBFT    Defocus Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 168

# 168    VBFT    Glass Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 168

# 168    VBFT    Glass Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 168

# 168    VBFT    Glass Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 168

# 168    VBFT    Glass Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 168

# 168    VBFT    Glass Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 168

# 168    VBFT    Motion Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 168

# 168    VBFT    Motion Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 168

# 168    VBFT    Motion Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 168

# 168    VBFT    Motion Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 168

# 168    VBFT    Motion Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 168

# 168    VBFT    Zoom Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 168

# 168    VBFT    Zoom Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 168

# 168    VBFT    Zoom Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 168

# 168    VBFT    Zoom Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 168

# 168    VBFT    Zoom Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 168

# 168    VBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 168

# 168    VBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 168

# 168    VBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 168

# 168    VBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 168

# 168    VBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 168

# 168    VBFT    Snow 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 168

# 168    VBFT    Snow 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 168

# 168    VBFT    Snow 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 168

# 168    VBFT    Snow 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 168

# 168    VBFT    Snow 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 168

# 168    VBFT    Frost 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 168

# 168    VBFT    Frost 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 168

# 168    VBFT    Frost 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 168

# 168    VBFT    Frost 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 168

# 168    VBFT    Frost 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 168

# 168    VBFT    Fog 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 168

# 168    VBFT    Fog 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 168

# 168    VBFT    Fog 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 168

# 168    VBFT    Fog 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 168

# 168    VBFT    Fog 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 168

# 168    VBFT    Spatter 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 168

# 168    VBFT    Spatter 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 168

# 168    VBFT    Spatter 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 168

# 168    VBFT    Spatter 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 168

# 168    VBFT    Spatter 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 168

# 168    VBFT    Brightness 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 168

# 168    VBFT    Brightness 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 168

# 168    VBFT    Brightness 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 168

# 168    VBFT    Brightness 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 168

# 168    VBFT    Brightness 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 168

# 168    VBFT    Contrast 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 168

# 168    VBFT    Contrast 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 168

# 168    VBFT    Contrast 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 168

# 168    VBFT    Contrast 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 168

# 168    VBFT    Contrast 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 168

# 168    VBFT    Elastic Transform 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 168

# 168    VBFT    Elastic Transform 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 168

# 168    VBFT    Elastic Transform 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 168

# 168    VBFT    Elastic Transform 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 168

# 168    VBFT    Elastic Transform 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 168

# 168    VBFT    Pixelate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 168

# 168    VBFT    Pixelate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 168

# 168    VBFT    Pixelate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 168

# 168    VBFT    Pixelate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 168

# 168    VBFT    Pixelate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 168

# 168    VBFT    JPEG 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 168

# 168    VBFT    JPEG 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 168

# 168    VBFT    JPEG 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 168

# 168    VBFT    JPEG 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 168

# 168    VBFT    JPEG 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 168

# 168    VBFT    Saturate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 168

# 168    VBFT    Saturate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 168

# 168    VBFT    Saturate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 168

# 168    VBFT    Saturate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 168

# 168    VBFT    Saturate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.4.32.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 168

# 210    RBFT    test
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -s 210

# 210    VBFT    test
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -s 210

# 210    RBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 210

# 210    RBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 210

# 210    RBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 210

# 210    RBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 210

# 210    RBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 210

# 210    RBFT    Shot Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 210

# 210    RBFT    Shot Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 210

# 210    RBFT    Shot Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 210

# 210    RBFT    Shot Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 210

# 210    RBFT    Shot Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 210

# 210    RBFT    Impulse Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 210

# 210    RBFT    Impulse Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 210

# 210    RBFT    Impulse Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 210

# 210    RBFT    Impulse Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 210

# 210    RBFT    Impulse Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 210

# 210    RBFT    Speckle Noise 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 210

# 210    RBFT    Speckle Noise 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 210

# 210    RBFT    Speckle Noise 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 210

# 210    RBFT    Speckle Noise 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 210

# 210    RBFT    Speckle Noise 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 210

# 210    RBFT    Defocus Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 210

# 210    RBFT    Defocus Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 210

# 210    RBFT    Defocus Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 210

# 210    RBFT    Defocus Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 210

# 210    RBFT    Defocus Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 210

# 210    RBFT    Glass Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 210

# 210    RBFT    Glass Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 210

# 210    RBFT    Glass Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 210

# 210    RBFT    Glass Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 210

# 210    RBFT    Glass Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 210

# 210    RBFT    Motion Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 210

# 210    RBFT    Motion Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 210

# 210    RBFT    Motion Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 210

# 210    RBFT    Motion Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 210

# 210    RBFT    Motion Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 210

# 210    RBFT    Zoom Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 210

# 210    RBFT    Zoom Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 210

# 210    RBFT    Zoom Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 210

# 210    RBFT    Zoom Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 210

# 210    RBFT    Zoom Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 210

# 210    RBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 210

# 210    RBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 210

# 210    RBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 210

# 210    RBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 210

# 210    RBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 210

# 210    RBFT    Snow 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 210

# 210    RBFT    Snow 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 210

# 210    RBFT    Snow 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 210

# 210    RBFT    Snow 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 210

# 210    RBFT    Snow 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 210

# 210    RBFT    Frost 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 210

# 210    RBFT    Frost 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 210

# 210    RBFT    Frost 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 210

# 210    RBFT    Frost 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 210

# 210    RBFT    Frost 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 210

# 210    RBFT    Fog 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 210

# 210    RBFT    Fog 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 210

# 210    RBFT    Fog 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 210

# 210    RBFT    Fog 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 210

# 210    RBFT    Fog 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 210

# 210    RBFT    Spatter 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 210

# 210    RBFT    Spatter 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 210

# 210    RBFT    Spatter 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 210

# 210    RBFT    Spatter 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 210

# 210    RBFT    Spatter 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 210

# 210    RBFT    Brightness 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 210

# 210    RBFT    Brightness 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 210

# 210    RBFT    Brightness 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 210

# 210    RBFT    Brightness 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 210

# 210    RBFT    Brightness 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 210

# 210    RBFT    Contrast 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 210

# 210    RBFT    Contrast 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 210

# 210    RBFT    Contrast 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 210

# 210    RBFT    Contrast 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 210

# 210    RBFT    Contrast 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 210

# 210    RBFT    Elastic Transform 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 210

# 210    RBFT    Elastic Transform 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 210

# 210    RBFT    Elastic Transform 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 210

# 210    RBFT    Elastic Transform 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 210

# 210    RBFT    Elastic Transform 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 210

# 210    RBFT    Pixelate 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 210

# 210    RBFT    Pixelate 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 210

# 210    RBFT    Pixelate 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 210

# 210    RBFT    Pixelate 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 210

# 210    RBFT    Pixelate 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 210

# 210    RBFT    JPEG 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 210

# 210    RBFT    JPEG 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 210

# 210    RBFT    JPEG 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 210

# 210    RBFT    JPEG 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 210

# 210    RBFT    JPEG 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 210

# 210    RBFT    Saturate 1
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 210

# 210    RBFT    Saturate 2
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 210

# 210    RBFT    Saturate 3
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 210

# 210    RBFT    Saturate 4
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 210

# 210    RBFT    Saturate 5
./test_decomposed.py -r baseline.resnet152.2023.6.8.18.33.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 210

# 210    VBFT    Gaussian Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_1/ -s 210

# 210    VBFT    Gaussian Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_2/ -s 210

# 210    VBFT    Gaussian Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_3/ -s 210

# 210    VBFT    Gaussian Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_4/ -s 210

# 210    VBFT    Gaussian Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 210

# 210    VBFT    Shot Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_1/ -s 210

# 210    VBFT    Shot Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_2/ -s 210

# 210    VBFT    Shot Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_3/ -s 210

# 210    VBFT    Shot Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_4/ -s 210

# 210    VBFT    Shot Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 210

# 210    VBFT    Impulse Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_1/ -s 210

# 210    VBFT    Impulse Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_2/ -s 210

# 210    VBFT    Impulse Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_3/ -s 210

# 210    VBFT    Impulse Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_4/ -s 210

# 210    VBFT    Impulse Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 210

# 210    VBFT    Speckle Noise 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_1/ -s 210

# 210    VBFT    Speckle Noise 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_2/ -s 210

# 210    VBFT    Speckle Noise 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_3/ -s 210

# 210    VBFT    Speckle Noise 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_4/ -s 210

# 210    VBFT    Speckle Noise 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 210

# 210    VBFT    Defocus Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_1/ -s 210

# 210    VBFT    Defocus Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_2/ -s 210

# 210    VBFT    Defocus Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_3/ -s 210

# 210    VBFT    Defocus Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_4/ -s 210

# 210    VBFT    Defocus Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 210

# 210    VBFT    Glass Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_1/ -s 210

# 210    VBFT    Glass Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_2/ -s 210

# 210    VBFT    Glass Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_3/ -s 210

# 210    VBFT    Glass Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_4/ -s 210

# 210    VBFT    Glass Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 210

# 210    VBFT    Motion Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_1/ -s 210

# 210    VBFT    Motion Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_2/ -s 210

# 210    VBFT    Motion Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_3/ -s 210

# 210    VBFT    Motion Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_4/ -s 210

# 210    VBFT    Motion Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 210

# 210    VBFT    Zoom Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_1/ -s 210

# 210    VBFT    Zoom Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_2/ -s 210

# 210    VBFT    Zoom Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_3/ -s 210

# 210    VBFT    Zoom Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_4/ -s 210

# 210    VBFT    Zoom Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 210

# 210    VBFT    Gaussian Blur 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_1/ -s 210

# 210    VBFT    Gaussian Blur 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_2/ -s 210

# 210    VBFT    Gaussian Blur 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_3/ -s 210

# 210    VBFT    Gaussian Blur 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_4/ -s 210

# 210    VBFT    Gaussian Blur 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 210

# 210    VBFT    Snow 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_1/ -s 210

# 210    VBFT    Snow 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_2/ -s 210

# 210    VBFT    Snow 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_3/ -s 210

# 210    VBFT    Snow 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_4/ -s 210

# 210    VBFT    Snow 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 210

# 210    VBFT    Frost 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_1/ -s 210

# 210    VBFT    Frost 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_2/ -s 210

# 210    VBFT    Frost 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_3/ -s 210

# 210    VBFT    Frost 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_4/ -s 210

# 210    VBFT    Frost 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 210

# 210    VBFT    Fog 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_1/ -s 210

# 210    VBFT    Fog 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_2/ -s 210

# 210    VBFT    Fog 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_3/ -s 210

# 210    VBFT    Fog 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_4/ -s 210

# 210    VBFT    Fog 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 210

# 210    VBFT    Spatter 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_1/ -s 210

# 210    VBFT    Spatter 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_2/ -s 210

# 210    VBFT    Spatter 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_3/ -s 210

# 210    VBFT    Spatter 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_4/ -s 210

# 210    VBFT    Spatter 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 210

# 210    VBFT    Brightness 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_1/ -s 210

# 210    VBFT    Brightness 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_2/ -s 210

# 210    VBFT    Brightness 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_3/ -s 210

# 210    VBFT    Brightness 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_4/ -s 210

# 210    VBFT    Brightness 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 210

# 210    VBFT    Contrast 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_1/ -s 210

# 210    VBFT    Contrast 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_2/ -s 210

# 210    VBFT    Contrast 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_3/ -s 210

# 210    VBFT    Contrast 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_4/ -s 210

# 210    VBFT    Contrast 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 210

# 210    VBFT    Elastic Transform 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_1/ -s 210

# 210    VBFT    Elastic Transform 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_2/ -s 210

# 210    VBFT    Elastic Transform 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_3/ -s 210

# 210    VBFT    Elastic Transform 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_4/ -s 210

# 210    VBFT    Elastic Transform 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 210

# 210    VBFT    Pixelate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_1/ -s 210

# 210    VBFT    Pixelate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_2/ -s 210

# 210    VBFT    Pixelate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_3/ -s 210

# 210    VBFT    Pixelate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_4/ -s 210

# 210    VBFT    Pixelate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 210

# 210    VBFT    JPEG 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_1/ -s 210

# 210    VBFT    JPEG 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_2/ -s 210

# 210    VBFT    JPEG 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_3/ -s 210

# 210    VBFT    JPEG 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_4/ -s 210

# 210    VBFT    JPEG 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_5/ -s 210

# 210    VBFT    Saturate 1
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_1/ -s 210

# 210    VBFT    Saturate 2
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_2/ -s 210

# 210    VBFT    Saturate 3
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_3/ -s 210

# 210    VBFT    Saturate 4
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_4/ -s 210

# 210    VBFT    Saturate 5
./test_decomposed.py -r baseline.vit_b_32.2023.6.8.11.6.Fractal-Define-XL-R2 -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 210



