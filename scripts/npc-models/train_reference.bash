#!/usr/bin/env bash

cd ../src

./train_reference.py -m "cbm" -b 256 -e 150 -s 42
