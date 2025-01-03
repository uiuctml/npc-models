#!/usr/bin/env bash

cd ../src

./counterfactual_attribute.py -rd "decomposed.resnet34_mtl.ft.42.2024.12.13.16.47.PowerEdge-R720" -rs "spn.pgd_discriminative.ft.42.2024.12.13.16.47.PowerEdge-R720" -s 42
