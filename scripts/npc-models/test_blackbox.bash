#!/usr/bin/env bash

cd "../../src/npc-models"

./test_blackbox.py -r "42.awa2.blackbox.resnet34.2024.12.14.8.11.PowerEdge-R720" -s 42
./test_blackbox.py -r "" -s 52
./test_blackbox.py -r "" -s 62
./test_blackbox.py -r "" -s 72
./test_blackbox.py -r "" -s 82
