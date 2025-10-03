#!/usr/bin/env bash

cd ../src

./counterfactual.py -rd "decomposed.resnet34_mtl.ft.42.2024.12.13.16.41.Aurora-R11" -rs "spn.pgd_discriminative.ft.42.2024.12.13.16.41.Aurora-R11" -s 42
