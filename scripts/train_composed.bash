#!/usr/bin/env bash

cd ../src

./train_composed.py -m "vit_b_32_mtl" -o "pgd_discriminative" -wd "decomposed.vit_b_32_mtl.ft.42.2024.10.24.19.12.newton.best.tar" -ws "spn.cccp_generative.nft.42.2024.10.24.18.12.Aurora-R11.best.tar" -fd 1 -fs 1 -r 0 -b 384 -e 100 -s 42
