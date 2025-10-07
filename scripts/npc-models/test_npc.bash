#!/usr/bin/env bash

cd "../../src/npc-models"

./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11" -p "42.awa2.pc.pgd.2024.12.13.16.41.Aurora-R11" -s 42
./test_npc.py -r "" -p "" -s 52
./test_npc.py -r "" -p "" -s 62
./test_npc.py -r "" -p "" -s 72
./test_npc.py -r "" -p "" -s 82

./test_npc.py -r "" -s 42
./test_npc.py -r "" -s 52
./test_npc.py -r "" -s 62
./test_npc.py -r "" -s 72
./test_npc.py -r "" -s 82
