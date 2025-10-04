#!/usr/bin/env bash

cd ../src

./counterfactual.py -r "42.decomposed.resnet34_mtl.2024.12.13.16.41.Aurora-R11" -p "42.spn.pgd.2024.12.13.16.41.Aurora-R11" -s 42
