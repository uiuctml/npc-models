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
./train_baseline.py -m "abm" -c 0 -l 1 -b 256 -e 150 -s 42
./train_baseline.py -m "abm" -c 0 -l 1 -b 256 -e 150 -s 52
./train_baseline.py -m "abm" -c 0 -l 1 -b 256 -e 150 -s 62
./train_baseline.py -m "abm" -c 0 -l 1 -b 256 -e 150 -s 72
./train_baseline.py -m "abm" -c 0 -l 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Original]
./train_baseline.py -m "cbm" -c 0 -l 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 0 -l 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 0 -l 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 0 -l 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 0 -l 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Hybrid, Supervised]
./train_baseline.py -m "cbm" -c 1 -l 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 1 -l 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 1 -l 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 1 -l 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 1 -l 1 -b 256 -e 150 -s 82

# Concept Bottleneck Model (CBM) [Hybrid, Unsupervised]
./train_baseline.py -m "cbm" -c 1 -l 0 -b 256 -e 150 -s 42
./train_baseline.py -m "cbm" -c 1 -l 0 -b 256 -e 150 -s 52
./train_baseline.py -m "cbm" -c 1 -l 0 -b 256 -e 150 -s 62
./train_baseline.py -m "cbm" -c 1 -l 0 -b 256 -e 150 -s 72
./train_baseline.py -m "cbm" -c 1 -l 0 -b 256 -e 150 -s 82

# Concept Embedding Model (CEM)
./train_baseline.py -m "cem" -c 0 -l 1 -b 256 -e 150 -s 42
./train_baseline.py -m "cem" -c 0 -l 1 -b 256 -e 150 -s 52
./train_baseline.py -m "cem" -c 0 -l 1 -b 256 -e 150 -s 62
./train_baseline.py -m "cem" -c 0 -l 1 -b 256 -e 150 -s 72
./train_baseline.py -m "cem" -c 0 -l 1 -b 256 -e 150 -s 82

# Deep Concept Reasoner (DCR)
./train_baseline.py -m "dcr" -c 0 -l 0 -w "42.awa2.baseline.cem.2025.10.23.14.26.Fractal-Define-R5-TV.best.zip" -b 256 -e 3000 -s 42
./train_baseline.py -m "dcr" -c 0 -l 0 -w "52.awa2.baseline.cem.2025.10.23.19.38.Fractal-Define-R5-TV.best.zip" -b 256 -e 3000 -s 52
./train_baseline.py -m "dcr" -c 0 -l 0 -w "62.awa2.baseline.cem.2025.10.24.0.46.Fractal-Define-R5-TV.best.zip" -b 256 -e 3000 -s 62
./train_baseline.py -m "dcr" -c 0 -l 0 -w "72.awa2.baseline.cem.2025.10.24.5.55.Fractal-Define-R5-TV.best.zip" -b 256 -e 3000 -s 72
./train_baseline.py -m "dcr" -c 0 -l 0 -w "82.awa2.baseline.cem.2025.10.24.11.4.Fractal-Define-R5-TV.best.zip" -b 256 -e 3000 -s 82
./train_baseline.py -m "dcr" -c 0 -l 0 -w "42.celeba.baseline.cem.2025.10.23.14.13.Fractal-Define-XL-R2.best.zip" -b 256 -e 3000 -s 42
./train_baseline.py -m "dcr" -c 0 -l 0 -w "52.celeba.baseline.cem.2025.10.24.6.41.Fractal-Define-XL-R2.best.zip" -b 256 -e 3000 -s 52
./train_baseline.py -m "dcr" -c 0 -l 0 -w "62.celeba.baseline.cem.2025.10.25.1.49.Fractal-Define-XL-R2.best.zip" -b 256 -e 3000 -s 62
./train_baseline.py -m "dcr" -c 0 -l 0 -w "72.celeba.baseline.cem.2025.10.25.19.57.Fractal-Define-XL-R2.best.zip" -b 256 -e 3000 -s 72
./train_baseline.py -m "dcr" -c 0 -l 0 -w "82.celeba.baseline.cem.2025.10.26.13.49.Fractal-Define-XL-R2.best.zip" -b 256 -e 3000 -s 82
./train_baseline.py -m "dcr" -c 0 -l 0 -w "42.gtsrb.baseline.cem.2025.10.23.14.37.PowerEdge-R720.best.zip" -b 256 -e 3000 -s 42
./train_baseline.py -m "dcr" -c 0 -l 0 -w "52.gtsrb.baseline.cem.2025.10.23.19.24.PowerEdge-R720.best.zip" -b 256 -e 3000 -s 52
./train_baseline.py -m "dcr" -c 0 -l 0 -w "62.gtsrb.baseline.cem.2025.10.24.0.7.PowerEdge-R720.best.zip" -b 256 -e 3000 -s 62
./train_baseline.py -m "dcr" -c 0 -l 0 -w "72.gtsrb.baseline.cem.2025.10.24.4.55.PowerEdge-R720.best.zip" -b 256 -e 3000 -s 72
./train_baseline.py -m "dcr" -c 0 -l 0 -w "82.gtsrb.baseline.cem.2025.10.24.9.37.PowerEdge-R720.best.zip" -b 256 -e 3000 -s 82
./train_baseline.py -m "dcr" -c 0 -l 0 -w "42.mnist.baseline.cem.2025.10.24.10.12.Aurora-R11.best.zip" -b 256 -e 3000 -s 42
./train_baseline.py -m "dcr" -c 0 -l 0 -w "52.mnist.baseline.cem.2025.10.24.12.10.Aurora-R11.best.zip" -b 256 -e 3000 -s 52
./train_baseline.py -m "dcr" -c 0 -l 0 -w "62.mnist.baseline.cem.2025.10.24.14.8.Aurora-R11.best.zip" -b 256 -e 3000 -s 62
./train_baseline.py -m "dcr" -c 0 -l 0 -w "72.mnist.baseline.cem.2025.10.24.16.4.Aurora-R11.best.zip" -b 256 -e 3000 -s 72
./train_baseline.py -m "dcr" -c 0 -l 0 -w "82.mnist.baseline.cem.2025.10.24.18.1.Aurora-R11.best.zip" -b 256 -e 3000 -s 82
