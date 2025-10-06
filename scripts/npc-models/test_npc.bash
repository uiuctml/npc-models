#!/usr/bin/env bash

cd "../../src/npc-models"

./test_npc.py -r "42.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11" -p "42.pc.pgd.2024.12.13.16.41.Aurora-R11" -s 42
