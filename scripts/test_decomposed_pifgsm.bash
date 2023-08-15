#!/usr/bin/env bash

cd ../src

# 42    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_0.pt -s 42 -a 1

# 42    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_1.pt -s 42 -a 1

# 42    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_2.pt -s 42 -a 1

# 42    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_3.pt -s 42 -a 1

# 42    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.4.23.42.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_0.pt -s 42 -a 1

# 42    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_1.pt -s 42 -a 1

# 42    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_2.pt -s 42 -a 1

# 42    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.4.23.42.euler.best.tar_task_3.pt -s 42 -a 1

# 42    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.4.23.41.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_0.pt -s 42 -a 1

# 42    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_1.pt -s 42 -a 1

# 42    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_2.pt -s 42 -a 1

# 42    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_3.pt -s 42 -a 1

# 42    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.5.13.15.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_0.pt -s 42 -a 1

# 42    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_1.pt -s 42 -a 1

# 42    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_2.pt -s 42 -a 1

# 42    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.5.13.15.euler.best.tar_task_3.pt -s 42 -a 1

# 42    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.5.13.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    CRMFT    CRMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_0.pt -s 42 -a 1

# 42    CRMFT    CRMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_1.pt -s 42 -a 1

# 42    CRMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_2.pt -s 42 -a 1

# 42    CRMFT    CRMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_3.pt -s 42 -a 1

# 42    CRMFT    CVMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_0.pt -s 42 -a 1

# 42    CRMFT    CVMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_1.pt -s 42 -a 1

# 42    CRMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_2.pt -s 42 -a 1

# 42    CRMFT    CVMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_3.pt -s 42 -a 1

# 42    CVMFT    CRMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_0.pt -s 42 -a 1

# 42    CVMFT    CRMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_1.pt -s 42 -a 1

# 42    CVMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_2.pt -s 42 -a 1

# 42    CVMFT    CRMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.42.2023.8.12.18.40.euler.best.tar_task_3.pt -s 42 -a 1

# 42    CVMFT    CVMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_0.pt -s 42 -a 1

# 42    CVMFT    CVMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_1.pt -s 42 -a 1

# 42    CVMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_2.pt -s 42 -a 1

# 42    CVMFT    CVMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.42.2023.8.12.21.44.newton.best.tar_task_3.pt -s 42 -a 1

# 42    CRMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_0.pt -s 42 -a 1

# 42    CRMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_1.pt -s 42 -a 1

# 42    CRMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_2.pt -s 42 -a 1

# 42    CRMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_3.pt -s 42 -a 1

# 42    CRMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    CRMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    CRMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    CRMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 42    CVMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_0.pt -s 42 -a 1

# 42    CVMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_1.pt -s 42 -a 1

# 42    CVMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_2.pt -s 42 -a 1

# 42    CVMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.42.2023.8.13.3.8.euler.best.tar_task_3.pt -s 42 -a 1

# 42    CVMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 42 -a 1

# 42    CVMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 42 -a 1

# 42    CVMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 42 -a 1

# 42    CVMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.42.2023.8.12.20.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 42 -a 1

# 84    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_0.pt -s 84 -a 1

# 84    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_1.pt -s 84 -a 1

# 84    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_2.pt -s 84 -a 1

# 84    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_3.pt -s 84 -a 1

# 84    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.1.7.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_0.pt -s 84 -a 1

# 84    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_1.pt -s 84 -a 1

# 84    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_2.pt -s 84 -a 1

# 84    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.1.7.euler.best.tar_task_3.pt -s 84 -a 1

# 84    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.8.17.39.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_0.pt -s 84 -a 1

# 84    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_1.pt -s 84 -a 1

# 84    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_2.pt -s 84 -a 1

# 84    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_3.pt -s 84 -a 1

# 84    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.14.40.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_0.pt -s 84 -a 1

# 84    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_1.pt -s 84 -a 1

# 84    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_2.pt -s 84 -a 1

# 84    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.14.40.euler.best.tar_task_3.pt -s 84 -a 1

# 84    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.22.42.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    CRMFT    CRMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CRMFT    CRMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CRMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CRMFT    CRMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CRMFT    CVMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CRMFT    CVMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CRMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CRMFT    CVMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CVMFT    CRMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CVMFT    CRMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CVMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CVMFT    CRMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.84.2023.8.12.19.54.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CVMFT    CVMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CVMFT    CVMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CVMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CVMFT    CVMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.84.2023.8.13.8.53.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CRMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CRMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CRMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CRMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CRMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    CRMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    CRMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    CRMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 84    CVMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_0.pt -s 84 -a 1

# 84    CVMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_1.pt -s 84 -a 1

# 84    CVMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_2.pt -s 84 -a 1

# 84    CVMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.84.2023.8.14.4.14.newton.best.tar_task_3.pt -s 84 -a 1

# 84    CVMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 84 -a 1

# 84    CVMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 84 -a 1

# 84    CVMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 84 -a 1

# 84    CVMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.84.2023.8.13.2.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 84 -a 1

# 126    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_0.pt -s 126 -a 1

# 126    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_1.pt -s 126 -a 1

# 126    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_2.pt -s 126 -a 1

# 126    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_3.pt -s 126 -a 1

# 126    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.9.13.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_0.pt -s 126 -a 1

# 126    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_1.pt -s 126 -a 1

# 126    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_2.pt -s 126 -a 1

# 126    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.9.13.23.euler.best.tar_task_3.pt -s 126 -a 1

# 126    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.6.46.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_0.pt -s 126 -a 1

# 126    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_1.pt -s 126 -a 1

# 126    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_2.pt -s 126 -a 1

# 126    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_3.pt -s 126 -a 1

# 126    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.11.20.36.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_0.pt -s 126 -a 1

# 126    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_1.pt -s 126 -a 1

# 126    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_2.pt -s 126 -a 1

# 126    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.11.20.36.euler.best.tar_task_3.pt -s 126 -a 1

# 126    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.6.34.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    CRMFT    CRMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_0.pt -s 126 -a 1

# 126    CRMFT    CRMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_1.pt -s 126 -a 1

# 126    CRMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_2.pt -s 126 -a 1

# 126    CRMFT    CRMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_3.pt -s 126 -a 1

# 126    CRMFT    CVMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    CRMFT    CVMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    CRMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    CRMFT    CVMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    CVMFT    CRMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_0.pt -s 126 -a 1

# 126    CVMFT    CRMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_1.pt -s 126 -a 1

# 126    CVMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_2.pt -s 126 -a 1

# 126    CVMFT    CRMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.126.2023.8.13.3.57.newton.best.tar_task_3.pt -s 126 -a 1

# 126    CVMFT    CVMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    CVMFT    CVMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    CVMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    CVMFT    CVMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.126.2023.8.11.19.14.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    CRMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_0.pt -s 126 -a 1

# 126    CRMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_1.pt -s 126 -a 1

# 126    CRMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_2.pt -s 126 -a 1

# 126    CRMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_3.pt -s 126 -a 1

# 126    CRMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    CRMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    CRMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    CRMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 126    CVMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_0.pt -s 126 -a 1

# 126    CVMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_1.pt -s 126 -a 1

# 126    CVMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_2.pt -s 126 -a 1

# 126    CVMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.126.2023.8.14.8.7.newton.best.tar_task_3.pt -s 126 -a 1

# 126    CVMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_0.pt -s 126 -a 1

# 126    CVMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_1.pt -s 126 -a 1

# 126    CVMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_2.pt -s 126 -a 1

# 126    CVMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.126.2023.8.13.8.15.Fractal-Define-XL-R2.best.tar_task_3.pt -s 126 -a 1

# 168    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_0.pt -s 168 -a 1

# 168    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_1.pt -s 168 -a 1

# 168    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_2.pt -s 168 -a 1

# 168    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_3.pt -s 168 -a 1

# 168    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.1.45.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_0.pt -s 168 -a 1

# 168    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_1.pt -s 168 -a 1

# 168    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_2.pt -s 168 -a 1

# 168    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.1.45.euler.best.tar_task_3.pt -s 168 -a 1

# 168    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.9.20.2.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_0.pt -s 168 -a 1

# 168    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_1.pt -s 168 -a 1

# 168    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_2.pt -s 168 -a 1

# 168    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_3.pt -s 168 -a 1

# 168    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.2.29.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_0.pt -s 168 -a 1

# 168    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_1.pt -s 168 -a 1

# 168    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_2.pt -s 168 -a 1

# 168    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.2.29.euler.best.tar_task_3.pt -s 168 -a 1

# 168    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.14.17.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    CRMFT    CRMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_0.pt -s 168 -a 1

# 168    CRMFT    CRMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_1.pt -s 168 -a 1

# 168    CRMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_2.pt -s 168 -a 1

# 168    CRMFT    CRMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_3.pt -s 168 -a 1

# 168    CRMFT    CVMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    CRMFT    CVMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    CRMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    CRMFT    CVMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    CVMFT    CRMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_0.pt -s 168 -a 1

# 168    CVMFT    CRMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_1.pt -s 168 -a 1

# 168    CVMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_2.pt -s 168 -a 1

# 168    CVMFT    CRMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.168.2023.8.13.12.4.newton.best.tar_task_3.pt -s 168 -a 1

# 168    CVMFT    CVMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    CVMFT    CVMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    CVMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    CVMFT    CVMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.168.2023.8.12.3.36.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    CRMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_0.pt -s 168 -a 1

# 168    CRMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_1.pt -s 168 -a 1

# 168    CRMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_2.pt -s 168 -a 1

# 168    CRMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_3.pt -s 168 -a 1

# 168    CRMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    CRMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    CRMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    CRMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 168    CVMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_0.pt -s 168 -a 1

# 168    CVMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_1.pt -s 168 -a 1

# 168    CVMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_2.pt -s 168 -a 1

# 168    CVMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.168.2023.8.14.12.1.newton.best.tar_task_3.pt -s 168 -a 1

# 168    CVMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_0.pt -s 168 -a 1

# 168    CVMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_1.pt -s 168 -a 1

# 168    CVMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_2.pt -s 168 -a 1

# 168    CVMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.168.2023.8.13.14.13.Fractal-Define-XL-R2.best.tar_task_3.pt -s 168 -a 1

# 210    RMFT    RMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_0.pt -s 210 -a 1

# 210    RMFT    RMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_1.pt -s 210 -a 1

# 210    RMFT    RMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_2.pt -s 210 -a 1

# 210    RMFT    RMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_3.pt -s 210 -a 1

# 210    RMFT    VMFT Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    RMFT    VMFT Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    RMFT    VMFT Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    RMFT    VMFT Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.10.14.3.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    VMFT    RMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_0.pt -s 210 -a 1

# 210    VMFT    RMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_1.pt -s 210 -a 1

# 210    VMFT    RMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_2.pt -s 210 -a 1

# 210    VMFT    RMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.10.14.3.euler.best.tar_task_3.pt -s 210 -a 1

# 210    VMFT    VMFT Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    VMFT    VMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    VMFT    VMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    VMFT    VMFT Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.10.9.16.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    RMFTL    RMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_0.pt -s 210 -a 1

# 210    RMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_1.pt -s 210 -a 1

# 210    RMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_2.pt -s 210 -a 1

# 210    RMFTL    RMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_3.pt -s 210 -a 1

# 210    RMFTL    VMFTL Color
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    RMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    RMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    RMFTL    VMFTL Text
./test_decomposed.py -r decomposed.resnet152_mtl.2023.6.12.8.23.euler -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    VMFTL    RMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_0.pt -s 210 -a 1

# 210    VMFTL    RMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_1.pt -s 210 -a 1

# 210    VMFTL    RMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_2.pt -s 210 -a 1

# 210    VMFTL    RMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet152_mtl.2023.6.12.8.23.euler.best.tar_task_3.pt -s 210 -a 1

# 210    VMFTL    VMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    VMFTL    VMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    VMFTL    VMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    VMFTL    VMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_mtl.2023.6.11.22.0.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    CRMFT    CRMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_0.pt -s 210 -a 1

# 210    CRMFT    CRMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_1.pt -s 210 -a 1

# 210    CRMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_2.pt -s 210 -a 1

# 210    CRMFT    CRMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_3.pt -s 210 -a 1

# 210    CRMFT    CVMFT Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    CRMFT    CVMFT Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    CRMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    CRMFT    CVMFT Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    CVMFT    CRMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_0.pt -s 210 -a 1

# 210    CVMFT    CRMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_1.pt -s 210 -a 1

# 210    CVMFT    CRMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_2.pt -s 210 -a 1

# 210    CVMFT    CRMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ft.210.2023.8.13.20.10.newton.best.tar_task_3.pt -s 210 -a 1

# 210    CVMFT    CVMFT Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    CVMFT    CVMFT Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    CVMFT    CVMFT Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    CVMFT    CVMFT Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ft.210.2023.8.12.11.56.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    CRMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_0.pt -s 210 -a 1

# 210    CRMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_1.pt -s 210 -a 1

# 210    CRMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_2.pt -s 210 -a 1

# 210    CRMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_3.pt -s 210 -a 1

# 210    CRMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    CRMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    CRMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    CRMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1

# 210    CVMFTL    CRMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_0.pt -s 210 -a 1

# 210    CVMFTL    CRMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_1.pt -s 210 -a 1

# 210    CVMFTL    CRMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_2.pt -s 210 -a 1

# 210    CVMFTL    CRMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.resnet101_clip_mtl.ftl.210.2023.8.14.14.50.newton.best.tar_task_3.pt -s 210 -a 1

# 210    CVMFTL    CVMFTL Color
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_0.pt -s 210 -a 1

# 210    CVMFTL    CVMFTL Shape
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_1.pt -s 210 -a 1

# 210    CVMFTL    CVMFTL Symbol
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_2.pt -s 210 -a 1

# 210    CVMFTL    CVMFTL Text
./test_decomposed.py -r decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2 -i ../../mapillary-dataset/images/split/adversarial/pifgsm/generated/decomposed.vit_b_32_clip_mtl.ftl.210.2023.8.13.20.9.Fractal-Define-XL-R2.best.tar_task_3.pt -s 210 -a 1