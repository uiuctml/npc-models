#!/usr/bin/env bash

cd "../../src/npc-models"

./train_npc.py -w "42.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11.best.zip" -c "42.pc.pgd.2024.12.13.16.41.Aurora-R11.best.zip" -b 256 -e 150 -s 42
