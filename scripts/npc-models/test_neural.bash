#!/usr/bin/env bash

cd "../../src/npc-models"

./test_neural.py -r "42.awa2.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11" -s 42
./test_neural.py -r "" -s 52
./test_neural.py -r "" -s 62
./test_neural.py -r "" -s 72
./test_neural.py -r "" -s 82
