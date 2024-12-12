#!/usr/bin/env bash

cd ../src

./train_baseline.py -m "resnet34" -b 256 -e 150 -s 42
