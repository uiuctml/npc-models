#!/usr/bin/env bash

##
# @file   train_blackbox.bash
# @author Simon Yu
# @date   12/12/2024
# @brief  Script for training blackbox models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./train_blackbox.py -b 256 -e 150 -s 42
./train_blackbox.py -b 256 -e 150 -s 52
./train_blackbox.py -b 256 -e 150 -s 62
./train_blackbox.py -b 256 -e 150 -s 72
./train_blackbox.py -b 256 -e 150 -s 82
