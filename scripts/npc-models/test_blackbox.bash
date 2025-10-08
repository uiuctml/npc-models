#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./test_blackbox.py -r "42.awa2.blackbox.resnet34.2024.12.14.8.11.PowerEdge-R720"
./test_blackbox.py -r "52.awa2.blackbox.resnet34.2024.12.23.6.4.Fractal-Define-R5-TV"
./test_blackbox.py -r "62.awa2.blackbox.resnet34.2024.12.23.21.47.Fractal-Define-R5-TV"
./test_blackbox.py -r "72.awa2.blackbox.resnet34.2024.12.24.13.30.Fractal-Define-R5-TV"
./test_blackbox.py -r "82.awa2.blackbox.resnet34.2024.12.25.5.12.Fractal-Define-R5-TV"
./test_blackbox.py -r "42.celeba.blackbox.resnet34.2024.12.20.17.14.PowerEdge-R720"
./test_blackbox.py -r "52.celeba.blackbox.resnet34.2024.12.29.23.45.PowerEdge-R720"
./test_blackbox.py -r "62.celeba.blackbox.resnet34.2024.12.30.5.14.Fractal-Define-R5-TV"
./test_blackbox.py -r "72.celeba.blackbox.resnet34.2024.12.30.5.16.PowerEdge-R720"
./test_blackbox.py -r "82.celeba.blackbox.resnet34.2024.12.31.9.22.Aurora-R11"
./test_blackbox.py -r "42.gtsrb.blackbox.resnet34.2024.12.12.15.13.PowerEdge-R720"
./test_blackbox.py -r "52.gtsrb.blackbox.resnet34.2024.12.23.4.14.Fractal-Define-XL-R2"
./test_blackbox.py -r "62.gtsrb.blackbox.resnet34.2024.12.23.18.11.Fractal-Define-XL-R2"
./test_blackbox.py -r "72.gtsrb.blackbox.resnet34.2024.12.24.8.6.Fractal-Define-XL-R2"
./test_blackbox.py -r "82.gtsrb.blackbox.resnet34.2024.12.24.22.2.Fractal-Define-XL-R2"
./test_blackbox.py -r "42.mnist.blackbox.resnet34.2024.12.12.15.14.PowerEdge-R720"
./test_blackbox.py -r "52.mnist.blackbox.resnet34.2024.12.23.2.49.Fractal-Define-XL-R2"
./test_blackbox.py -r "62.mnist.blackbox.resnet34.2024.12.23.15.1.Fractal-Define-XL-R2"
./test_blackbox.py -r "72.mnist.blackbox.resnet34.2024.12.24.3.13.Fractal-Define-XL-R2"
./test_blackbox.py -r "82.mnist.blackbox.resnet34.2024.12.24.15.25.Fractal-Define-XL-R2"
