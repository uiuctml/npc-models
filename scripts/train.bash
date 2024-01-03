#!/usr/bin/env bash

cd ../src

./train_baseline.py -m "resnet152_mtl" -b 384 -e 50 -c 0 -s 42
./train_baseline.py -m "vit_b_32_mtl" -b 384 -e 50 -c 0 -s 42

./train_decomposed.py -m "resnet152_mtl" -b 384 -e 50 -c 0 -s 42
./train_decomposed.py -m "vit_b_32_mtl" -b 384 -e 50 -c 0 -s 42
