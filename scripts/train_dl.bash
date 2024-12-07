#!/usr/bin/env bash

cd ../src

./train_dl_mtl.py -m "resnet34_mtl" -b 256 -e 150 -s 42
