#!/usr/bin/env bash

cd "../../src/npc-models"

./train_npc.py -w "42.awa2.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11.best.zip" -c "42.awa2.pc.pgd.2024.12.13.16.41.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "" -c "" -b 256 -e 150 -s 52
./train_npc.py -w "" -c "" -b 256 -e 150 -s 62
./train_npc.py -w "" -c "" -b 256 -e 150 -s 72
./train_npc.py -w "" -c "" -b 256 -e 150 -s 82

./train_npc.py -w "" -b 256 -e 150 -s 42
./train_npc.py -w "" -b 256 -e 150 -s 52
./train_npc.py -w "" -b 256 -e 150 -s 62
./train_npc.py -w "" -b 256 -e 150 -s 72
./train_npc.py -w "" -b 256 -e 150 -s 82
