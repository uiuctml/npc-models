#!/usr/bin/env bash

cd ../src

./train_composed.py -m "resnet34_mtl" -o "pgd_discriminative" -wd "decomposed.resnet34_mtl.ft.42.2024.12.13.16.41.Aurora-R11.best.tar" -ws "spn.pgd_discriminative.ft.42.2024.12.13.16.41.Aurora-R11.best.tar" -fd 1 -fs 1 -is 1 -r 0 -b 256 -e 150 -s 42
