#!/usr/bin/env bash

cd ../src

./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_noise_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/shot_noise_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/impulse_noise_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/speckle_noise_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/defocus_blur_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/glass_blur_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/motion_blur_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/zoom_blur_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/gaussian_blur_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/snow_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/frost_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/fog_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/spatter_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/brightness_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/contrast_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/elastic_transform_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/pixelate_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/jpeg_compression_5/ -s 42
./test_composed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -d ../../mapillary-dataset/images/split/corrupted/generated/test/saturate_5/ -s 42
