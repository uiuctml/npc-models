#!/usr/bin/env bash

cd "../../src/npc-models"

./test_pc.py -r "42.awa2.pc.pgd.2024.12.13.16.41.Aurora-R11" -s 42
./test_pc.py -r "" -s 52
./test_pc.py -r "" -s 62
./test_pc.py -r "" -s 72
./test_pc.py -r "" -s 82
