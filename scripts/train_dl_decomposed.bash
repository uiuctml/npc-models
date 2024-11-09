#!/usr/bin/env bash

cd ../src

./train_dl_mtl.py -m "vit_b_32_mtl" -b 384 -e 100 -c 0 -s 42
