#!/usr/bin/env bash

cd "../../src/npc-models"

./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11" -p "42.awa2.pc.pgd.2024.12.13.16.41.Aurora-R11"
./test_npc.py -r "" -p ""
./test_npc.py -r "" -p ""
./test_npc.py -r "" -p ""
./test_npc.py -r "" -p ""

./test_npc.py -r ""
./test_npc.py -r ""
./test_npc.py -r ""
./test_npc.py -r ""
./test_npc.py -r ""
