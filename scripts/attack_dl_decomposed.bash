#!/usr/bin/env bash

cd ../src

./attack_dl_decomposed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -a 0 -s 42
./attack_dl_decomposed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -a 1 -s 42
./attack_dl_decomposed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -a 2 -s 42
./attack_dl_decomposed.py -r decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -a 3 -s 42
