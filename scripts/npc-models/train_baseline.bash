#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./train_baseline.py -m "abm" -b 256 -e 150 -s 42
./train_baseline.py -m "abm" -b 256 -e 150 -s 52
./train_baseline.py -m "abm" -b 256 -e 150 -s 62
./train_baseline.py -m "abm" -b 256 -e 150 -s 72
./train_baseline.py -m "abm" -b 256 -e 150 -s 82
./train_baseline.py -m "cbm" -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -b 256 -e 150 -s 82
./train_baseline.py -m "cem" -b 256 -e 150 -s 42
./train_baseline.py -m "cem" -b 256 -e 150 -s 52
./train_baseline.py -m "cem" -b 256 -e 150 -s 62
./train_baseline.py -m "cem" -b 256 -e 150 -s 72
./train_baseline.py -m "cem" -b 256 -e 150 -s 82
./train_baseline.py -m "dcr" -b 256 -e 150 -s 42
./train_baseline.py -m "dcr" -b 256 -e 150 -s 52
./train_baseline.py -m "dcr" -b 256 -e 150 -s 62
./train_baseline.py -m "dcr" -b 256 -e 150 -s 72
./train_baseline.py -m "dcr" -b 256 -e 150 -s 82
