#!/usr/bin/env bash

cd ../src

./train_composed.py -m "mlp_set" -o "pgd_composed" -wd "decomposed.mlp_set.ft.42.2024.2.12.16.7.euler.best.tar" -fd 1 -fs 0 -r 1 -b 384 -e 100 -s 42
