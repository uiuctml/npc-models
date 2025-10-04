#!/usr/bin/env bash

cd ../src

./train_composed.py -w "42.decomposed.resnet34_mtl.2024.12.13.16.41.Aurora-R11.best.tar" -c "42.spn.pgd.2024.12.13.16.41.Aurora-R11.best.tar" -b 256 -e 150 -s 42
