#!/usr/bin/env bash

##
# @file   train_pc.bash
# @author Simon Yu
# @date   02/13/2024
# @brief  Script for training PC models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

./train_pc.py -e 50 -s 42
./train_pc.py -e 50 -s 52
./train_pc.py -e 50 -s 62
./train_pc.py -e 50 -s 72
./train_pc.py -e 50 -s 82
