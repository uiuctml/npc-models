#!/usr/bin/env bash

##
# @file   train_neural.bash
# @author Simon Yu
# @date   12/06/2024
# @brief  Script for training neural models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./train_neural.py -b 256 -e 150 -s 42
./train_neural.py -b 256 -e 150 -s 52
./train_neural.py -b 256 -e 150 -s 62
./train_neural.py -b 256 -e 150 -s 72
./train_neural.py -b 256 -e 150 -s 82
