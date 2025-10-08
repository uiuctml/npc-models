#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./train_pc.py -e 50 -s 42
./train_pc.py -e 50 -s 52
./train_pc.py -e 50 -s 62
./train_pc.py -e 50 -s 72
./train_pc.py -e 50 -s 82
