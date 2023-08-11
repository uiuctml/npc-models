#!/usr/bin/env bash

cd ../src

./train_baseline.py -f 1 -s 42
./train_baseline.py -f 1 -s 84
./train_baseline.py -f 1 -s 126
./train_baseline.py -f 1 -s 168
./train_baseline.py -f 1 -s 210

./train_decomposed.py -f 1 -s 42
./train_decomposed.py -f 1 -s 84
./train_decomposed.py -f 1 -s 126
./train_decomposed.py -f 1 -s 168
./train_decomposed.py -f 1 -s 210

./train_decomposed.py -f 0 -s 42
./train_decomposed.py -f 0 -s 84
./train_decomposed.py -f 0 -s 126
./train_decomposed.py -f 0 -s 168
./train_decomposed.py -f 0 -s 210
