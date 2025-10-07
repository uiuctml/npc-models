#!/usr/bin/env bash

cd "../../src/npc-models"

./train_neural.py -b 256 -e 150 -s 42
./train_neural.py -b 256 -e 150 -s 52
./train_neural.py -b 256 -e 150 -s 62
./train_neural.py -b 256 -e 150 -s 72
./train_neural.py -b 256 -e 150 -s 82
