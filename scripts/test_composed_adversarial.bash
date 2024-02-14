#!/usr/bin/env bash

cd ../src

./test_composed.py -rd decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -rs spn.cccp_offline.ft.42.2024.2.13.21.10.Aurora-R12 -d ../../gtsrb-dataset/images/split/adversarial/test/mlp_set_pgd_color/ -s 42
./test_composed.py -rd decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -rs spn.cccp_offline.ft.42.2024.2.13.21.10.Aurora-R12 -d ../../gtsrb-dataset/images/split/adversarial/test/mlp_set_pgd_shape/ -s 42
./test_composed.py -rd decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -rs spn.cccp_offline.ft.42.2024.2.13.21.10.Aurora-R12 -d ../../gtsrb-dataset/images/split/adversarial/test/mlp_set_pgd_symbol/ -s 42
./test_composed.py -rd decomposed.mlp_set.ft.42.2024.2.12.16.7.euler -rs spn.cccp_offline.ft.42.2024.2.13.21.10.Aurora-R12 -d ../../gtsrb-dataset/images/split/adversarial/test/mlp_set_pgd_text/ -s 42
