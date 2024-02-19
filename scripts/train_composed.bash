#!/usr/bin/env bash

cd ../src

./train_composed.py -m "mlp_set" -o "cccp_composed" -wd "decomposed.mlp_set.ft.42.2024.2.12.16.7.euler.best.tar" -ws "spn.cccp_offline.ft.42.2024.2.13.22.3.Aurora-R11.best.tar" -fd 1 -fs 0 -r 0 -b 384 -e 100 -s 42
