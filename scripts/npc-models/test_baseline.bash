#!/usr/bin/env bash

##
# @file   test_baseline.bash
# @author Simon Yu
# @date   10/03/2025
# @brief  Script for testing baseline models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

# Attribute Bottleneck Model (ABM)
./test_baseline.py -r "42.awa2.baseline.abm.2024.12.13.7.6.PowerEdge-R720" -c 0
./test_baseline.py -r "52.awa2.baseline.abm.2024.12.22.20.21.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "62.awa2.baseline.abm.2024.12.23.12.10.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "72.awa2.baseline.abm.2024.12.24.3.53.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "82.awa2.baseline.abm.2024.12.24.19.36.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "42.celeba.baseline.abm.2024.12.17.22.10.PowerEdge-R720" -c 0
./test_baseline.py -r "52.celeba.baseline.abm.2024.12.26.3.3.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "62.celeba.baseline.abm.2024.12.27.3.13.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "72.celeba.baseline.abm.2024.12.28.3.28.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "82.celeba.baseline.abm.2024.12.29.3.19.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "42.gtsrb.baseline.abm.2024.12.7.0.50.PowerEdge-R720" -c 0
./test_baseline.py -r "52.gtsrb.baseline.abm.2024.12.22.19.35.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "62.gtsrb.baseline.abm.2024.12.23.9.34.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "72.gtsrb.baseline.abm.2024.12.23.23.30.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "82.gtsrb.baseline.abm.2024.12.24.13.26.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "42.mnist.baseline.abm.2024.12.6.18.41.PowerEdge-R720" -c 0
./test_baseline.py -r "52.mnist.baseline.abm.2024.12.22.19.21.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "62.mnist.baseline.abm.2024.12.23.7.36.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "72.mnist.baseline.abm.2024.12.23.19.48.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "82.mnist.baseline.abm.2024.12.24.8.0.Fractal-Define-XL-R2" -c 0

# Concept Bottleneck Model (CBM) [Original]
./test_baseline.py -r "42.awa2.baseline.cbm.2024.12.13.0.6.PowerEdge-R720" -c 0
./test_baseline.py -r "52.awa2.baseline.cbm.2024.12.22.17.16.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "62.awa2.baseline.cbm.2024.12.23.9.7.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "72.awa2.baseline.cbm.2024.12.24.0.50.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "82.awa2.baseline.cbm.2024.12.24.16.33.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "42.celeba.baseline.cbm.2024.12.17.22.7.PowerEdge-R720" -c 0
./test_baseline.py -r "52.celeba.baseline.cbm.2024.12.26.3.1.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "62.celeba.baseline.cbm.2024.12.26.14.51.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "72.celeba.baseline.cbm.2024.12.27.2.41.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "82.celeba.baseline.cbm.2024.12.27.14.31.Fractal-Define-R5-TV" -c 0
./test_baseline.py -r "42.gtsrb.baseline.cbm.2024.12.7.0.52.PowerEdge-R720" -c 0
./test_baseline.py -r "52.gtsrb.baseline.cbm.2024.12.22.16.56.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "62.gtsrb.baseline.cbm.2024.12.23.6.54.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "72.gtsrb.baseline.cbm.2024.12.23.20.50.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "82.gtsrb.baseline.cbm.2024.12.24.10.45.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "42.mnist.baseline.cbm.2024.12.6.18.40.PowerEdge-R720" -c 0
./test_baseline.py -r "52.mnist.baseline.cbm.2024.12.22.16.58.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "62.mnist.baseline.cbm.2024.12.23.5.12.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "72.mnist.baseline.cbm.2024.12.23.17.24.Fractal-Define-XL-R2" -c 0
./test_baseline.py -r "82.mnist.baseline.cbm.2024.12.24.5.36.Fractal-Define-XL-R2" -c 0

# Concept Bottleneck Model (CBM) [Hybrid, Supervised]
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1

# Concept Bottleneck Model (CBM) [Hybrid, Unsupervised]
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1
./test_baseline.py -r "" -c 1

# Concept Embedding Model (CEM)
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0

# Deep Concept Reasoner (DCR)
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
./test_baseline.py -r "" -c 0
