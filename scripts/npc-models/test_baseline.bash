#!/usr/bin/env bash

cd "../../src/npc-models"

./test_baseline.py -r "42.awa2.baseline.cbm.2024.12.13.0.6.PowerEdge-R720" -s 42
./test_baseline.py -r "" -s 42
./test_baseline.py -r "" -s 42
./test_baseline.py -r "" -s 42

./test_baseline.py -r "" -s 52
./test_baseline.py -r "" -s 52
./test_baseline.py -r "" -s 52
./test_baseline.py -r "" -s 52

./test_baseline.py -r "" -s 62
./test_baseline.py -r "" -s 62
./test_baseline.py -r "" -s 62
./test_baseline.py -r "" -s 62

./test_baseline.py -r "" -s 72
./test_baseline.py -r "" -s 72
./test_baseline.py -r "" -s 72
./test_baseline.py -r "" -s 72

./test_baseline.py -r "" -s 82
./test_baseline.py -r "" -s 82
./test_baseline.py -r "" -s 82
./test_baseline.py -r "" -s 82
