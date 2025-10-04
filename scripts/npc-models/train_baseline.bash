#!/usr/bin/env bash

cd "../../src/npc-models"

./train_baseline.py -b 256 -e 150 -s 42
