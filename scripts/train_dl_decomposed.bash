#!/usr/bin/env bash

cd ../src

./train_dl_mlp.py -m "mlp_set" -b 384 -e 100 -c 0 -s 42
