#!/usr/bin/env bash

cd ../src

# 42    RMFT    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -s 42

# 42    VMFT    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -s 42

# 42    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_0.pt -s 42 -a 1

# 42    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_1.pt -s 42 -a 1

# 42    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_2.pt -s 42 -a 1

# 42    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_3.pt -s 42 -a 1

# 42    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_0.pt -s 42 -a 1

# 42    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_1.pt -s 42 -a 1

# 42    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_2.pt -s 42 -a 1

# 42    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_3.pt -s 42 -a 1

# 42    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    RMFTL    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -s 42

# 42    VMFTL    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -s 42

# 42    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_0.pt -s 42 -a 1

# 42    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_1.pt -s 42 -a 1

# 42    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_2.pt -s 42 -a 1

# 42    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_3.pt -s 42 -a 1

# 42    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_0.pt -s 42 -a 1

# 42    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_1.pt -s 42 -a 1

# 42    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_2.pt -s 42 -a 1

# 42    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_3.pt -s 42 -a 1

# 42    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 84    RMFT    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -s 84

# 84    VMFT    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -s 84

# 84    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_0.pt -s 84 -a 1

# 84    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_1.pt -s 84 -a 1

# 84    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_2.pt -s 84 -a 1

# 84    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_3.pt -s 84 -a 1

# 84    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_0.pt -s 84 -a 1

# 84    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_1.pt -s 84 -a 1

# 84    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_2.pt -s 84 -a 1

# 84    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_3.pt -s 84 -a 1

# 84    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    RMFTL    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -s 84

# 84    VMFTL    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -s 84

# 84    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_0.pt -s 84 -a 1

# 84    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_1.pt -s 84 -a 1

# 84    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_2.pt -s 84 -a 1

# 84    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_3.pt -s 84 -a 1

# 84    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_0.pt -s 84 -a 1

# 84    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_1.pt -s 84 -a 1

# 84    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_2.pt -s 84 -a 1

# 84    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_3.pt -s 84 -a 1

# 84    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 126    RMFT    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -s 126

# 126    VMFT    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -s 126

# 126    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_0.pt -s 126 -a 1

# 126    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_1.pt -s 126 -a 1

# 126    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_2.pt -s 126 -a 1

# 126    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_3.pt -s 126 -a 1

# 126    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_0.pt -s 126 -a 1

# 126    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_1.pt -s 126 -a 1

# 126    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_2.pt -s 126 -a 1

# 126    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_3.pt -s 126 -a 1

# 126    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    RMFTL    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -s 126

# 126    VMFTL    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -s 126

# 126    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_0.pt -s 126 -a 1

# 126    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_1.pt -s 126 -a 1

# 126    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_2.pt -s 126 -a 1

# 126    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_3.pt -s 126 -a 1

# 126    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_0.pt -s 126 -a 1

# 126    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_1.pt -s 126 -a 1

# 126    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_2.pt -s 126 -a 1

# 126    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_3.pt -s 126 -a 1

# 126    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 168    RMFT    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -s 168

# 168    VMFT    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -s 168

# 168    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_0.pt -s 168 -a 1

# 168    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_1.pt -s 168 -a 1

# 168    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_2.pt -s 168 -a 1

# 168    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_3.pt -s 168 -a 1

# 168    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_0.pt -s 168 -a 1

# 168    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_1.pt -s 168 -a 1

# 168    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_2.pt -s 168 -a 1

# 168    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_3.pt -s 168 -a 1

# 168    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    RMFTL    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -s 168

# 168    VMFTL    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -s 168

# 168    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_0.pt -s 168 -a 1

# 168    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_1.pt -s 168 -a 1

# 168    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_2.pt -s 168 -a 1

# 168    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_3.pt -s 168 -a 1

# 168    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_0.pt -s 168 -a 1

# 168    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_1.pt -s 168 -a 1

# 168    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_2.pt -s 168 -a 1

# 168    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_3.pt -s 168 -a 1

# 168    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 210    RMFT    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -s 210

# 210    VMFT    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -s 210

# 210    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_0.pt -s 210 -a 1

# 210    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_1.pt -s 210 -a 1

# 210    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_2.pt -s 210 -a 1

# 210    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_3.pt -s 210 -a 1

# 210    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_0.pt -s 210 -a 1

# 210    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_1.pt -s 210 -a 1

# 210    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_2.pt -s 210 -a 1

# 210    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_3.pt -s 210 -a 1

# 210    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    RMFTL    N/A 
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -s 210

# 210    VMFTL    N/A
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -s 210

# 210    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_0.pt -s 210 -a 1

# 210    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_1.pt -s 210 -a 1

# 210    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_2.pt -s 210 -a 1

# 210    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_3.pt -s 210 -a 1

# 210    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_0.pt -s 210 -a 1

# 210    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_1.pt -s 210 -a 1

# 210    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_2.pt -s 210 -a 1

# 210    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_3.pt -s 210 -a 1

# 210    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/pgd/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

