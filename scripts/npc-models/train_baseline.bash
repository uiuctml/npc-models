#!/usr/bin/env bash

##
# @file   train_baseline.bash
# @author Simon Yu
# @date   12/06/2024
# @brief  Script for training baseline models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

# Attribute Bottleneck Model (ABM)
./train_baseline.py -m "abm" -c 0 -w 1 -b 256 -e 150 -s 42
./train_baseline.py -m "abm" -c 0 -w 1 -b 256 -e 150 -s 52
./train_baseline.py -m "abm" -c 0 -w 1 -b 256 -e 150 -s 62
./train_baseline.py -m "abm" -c 0 -w 1 -b 256 -e 150 -s 72
./train_baseline.py -m "abm" -c 0 -w 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Original]
./train_baseline.py -m "cbm" -c 0 -w 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 0 -w 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 0 -w 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 0 -w 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 0 -w 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Hybrid, Supervised]
./train_baseline.py -m "cbm" -c 1 -w 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 1 -w 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 1 -w 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 1 -w 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 1 -w 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Hybrid, Unsupervised]
./train_baseline.py -m "cbm" -c 1 -w 0 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 1 -w 0 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 1 -w 0 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 1 -w 0 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 1 -w 0 -b 256 -e 150 -s 82

# Concept Embedding Model (CEM)
./train_baseline.py -m "cem" -c 0 -w 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cem" -c 0 -w 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cem" -c 0 -w 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cem" -c 0 -w 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cem" -c 0 -w 1 -b 256 -e 150 -s 82

# Deep Concept Reasoner (DCR)
./train_baseline.py -m "dcr" -c 0 -w 1 -b 256 -e 150 -s 42
./train_baseline.py -m "dcr" -c 0 -w 1 -b 256 -e 150 -s 52
./train_baseline.py -m "dcr" -c 0 -w 1 -b 256 -e 150 -s 62
./train_baseline.py -m "dcr" -c 0 -w 1 -b 256 -e 150 -s 72
./train_baseline.py -m "dcr" -c 0 -w 1 -b 256 -e 150 -s 82
