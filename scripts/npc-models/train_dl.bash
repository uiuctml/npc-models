#!/usr/bin/env bash

cd "../../src/npc-models"

./train_dl.py -b 256 -e 150 -s 42
