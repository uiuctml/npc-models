#!/usr/bin/env bash

##
# @file   test_npc.bash
# @author Simon Yu
# @date   11/09/2024
# @brief  Script for testing NPC models.
##

# Go to script directory
cd "$(dirname $0)"

# Go to source directory
cd "../../src/npc-models"

# Independently trained with data-driven PC
./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11" -p "42.awa2.pc.cccp.2024.12.12.19.55.Aurora-R11"
./test_npc.py -r "52.awa2.neural.resnet34mtl.2024.12.24.0.30.Aurora-R11" -p "52.awa2.pc.cccp.2024.12.22.15.16.Aurora-R11"
./test_npc.py -r "62.awa2.neural.resnet34mtl.2024.12.24.3.25.Aurora-R11" -p "62.awa2.pc.cccp.2024.12.22.15.17.Aurora-R11"
./test_npc.py -r "72.awa2.neural.resnet34mtl.2024.12.24.6.21.Aurora-R11" -p "72.awa2.pc.cccp.2024.12.22.15.17.Aurora-R11"
./test_npc.py -r "82.awa2.neural.resnet34mtl.2024.12.24.9.16.Aurora-R11" -p "82.awa2.pc.cccp.2024.12.22.15.18.Aurora-R11"
./test_npc.py -r "42.celeba.neural.resnet34mtl.2024.12.17.21.56.Aurora-R11" -p "42.celeba.pc.cccp.2024.12.17.21.53.Aurora-R11"
./test_npc.py -r "52.celeba.neural.resnet34mtl.2024.12.26.2.57.Fractal-Define-XL-R2" -p "52.celeba.pc.cccp.2024.12.22.15.24.Aurora-R11"
./test_npc.py -r "62.celeba.neural.resnet34mtl.2024.12.26.19.16.Fractal-Define-XL-R2" -p "62.celeba.pc.cccp.2024.12.22.15.26.Aurora-R11"
./test_npc.py -r "72.celeba.neural.resnet34mtl.2024.12.26.2.58.Fractal-Define-XL-R2" -p "72.celeba.pc.cccp.2024.12.22.15.28.Aurora-R11"
./test_npc.py -r "82.celeba.neural.resnet34mtl.2024.12.26.19.20.Fractal-Define-XL-R2" -p "82.celeba.pc.cccp.2024.12.22.15.29.Aurora-R11"
./test_npc.py -r "42.gtsrb.neural.resnet34mtl.2024.12.10.18.10.PowerEdge-R720" -p "42.gtsrb.pc.cccp.2024.12.11.16.22.Aurora-R11"
./test_npc.py -r "52.gtsrb.neural.resnet34mtl.2024.12.24.17.41.Aurora-R11" -p "52.gtsrb.pc.cccp.2024.12.22.15.33.Aurora-R11"
./test_npc.py -r "62.gtsrb.neural.resnet34mtl.2024.12.24.5.54.PowerEdge-R720" -p "62.gtsrb.pc.cccp.2024.12.22.15.33.Aurora-R11"
./test_npc.py -r "72.gtsrb.neural.resnet34mtl.2024.12.24.11.37.PowerEdge-R720" -p "72.gtsrb.pc.cccp.2024.12.22.15.34.Aurora-R11"
./test_npc.py -r "82.gtsrb.neural.resnet34mtl.2024.12.24.17.21.PowerEdge-R720" -p "82.gtsrb.pc.cccp.2024.12.22.15.34.Aurora-R11"
./test_npc.py -r "42.mnist.neural.resnet34mtl.2024.12.10.18.11.PowerEdge-R720" -p "42.mnist.pc.cccp.2024.12.11.16.23.Aurora-R11"
./test_npc.py -r "52.mnist.neural.resnet34mtl.2024.12.24.0.10.PowerEdge-R720" -p "52.mnist.pc.cccp.2024.12.22.15.53.Aurora-R11"
./test_npc.py -r "62.mnist.neural.resnet34mtl.2024.12.24.4.40.PowerEdge-R720" -p "62.mnist.pc.cccp.2024.12.22.15.54.Aurora-R11"
./test_npc.py -r "72.mnist.neural.resnet34mtl.2024.12.24.9.9.PowerEdge-R720" -p "72.mnist.pc.cccp.2024.12.22.15.55.Aurora-R11"
./test_npc.py -r "82.mnist.neural.resnet34mtl.2024.12.24.13.37.PowerEdge-R720" -p "82.mnist.pc.cccp.2024.12.22.15.56.Aurora-R11"

# Independently trained with knowledge-injected PC
# Update header.config_pc["file_path_pc"] to point to knowledge-injected PC
./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11"
./test_npc.py -r "52.awa2.neural.resnet34mtl.2024.12.24.0.30.Aurora-R11"
./test_npc.py -r "62.awa2.neural.resnet34mtl.2024.12.24.3.25.Aurora-R11"
./test_npc.py -r "72.awa2.neural.resnet34mtl.2024.12.24.6.21.Aurora-R11"
./test_npc.py -r "82.awa2.neural.resnet34mtl.2024.12.24.9.16.Aurora-R11"
./test_npc.py -r "42.celeba.neural.resnet34mtl.2024.12.17.21.56.Aurora-R11"
./test_npc.py -r "52.celeba.neural.resnet34mtl.2024.12.26.2.57.Fractal-Define-XL-R2"
./test_npc.py -r "62.celeba.neural.resnet34mtl.2024.12.26.19.16.Fractal-Define-XL-R2"
./test_npc.py -r "72.celeba.neural.resnet34mtl.2024.12.26.2.58.Fractal-Define-XL-R2"
./test_npc.py -r "82.celeba.neural.resnet34mtl.2024.12.26.19.20.Fractal-Define-XL-R2"
./test_npc.py -r "42.gtsrb.neural.resnet34mtl.2024.12.10.18.10.PowerEdge-R720"
./test_npc.py -r "52.gtsrb.neural.resnet34mtl.2024.12.24.17.41.Aurora-R11"
./test_npc.py -r "62.gtsrb.neural.resnet34mtl.2024.12.24.5.54.PowerEdge-R720"
./test_npc.py -r "72.gtsrb.neural.resnet34mtl.2024.12.24.11.37.PowerEdge-R720"
./test_npc.py -r "82.gtsrb.neural.resnet34mtl.2024.12.24.17.21.PowerEdge-R720"
./test_npc.py -r "42.mnist.neural.resnet34mtl.2024.12.10.18.11.PowerEdge-R720"
./test_npc.py -r "52.mnist.neural.resnet34mtl.2024.12.24.0.10.PowerEdge-R720"
./test_npc.py -r "62.mnist.neural.resnet34mtl.2024.12.24.4.40.PowerEdge-R720"
./test_npc.py -r "72.mnist.neural.resnet34mtl.2024.12.24.9.9.PowerEdge-R720"
./test_npc.py -r "82.mnist.neural.resnet34mtl.2024.12.24.13.37.PowerEdge-R720"

# Jointly optimized with data-driven PC
./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.13.16.41.Aurora-R11" -p "42.awa2.pc.pgd.2024.12.13.16.41.Aurora-R11"
./test_npc.py -r "52.awa2.neural.resnet34mtl.2024.12.29.0.44.Fractal-Define-XL-R2" -p "52.awa2.pc.pgd.2024.12.29.0.44.Fractal-Define-XL-R2"
./test_npc.py -r "62.awa2.neural.resnet34mtl.2024.12.29.4.38.Fractal-Define-XL-R2" -p "62.awa2.pc.pgd.2024.12.29.4.38.Fractal-Define-XL-R2"
./test_npc.py -r "72.awa2.neural.resnet34mtl.2024.12.29.8.36.Fractal-Define-XL-R2" -p "72.awa2.pc.pgd.2024.12.29.8.36.Fractal-Define-XL-R2"
./test_npc.py -r "82.awa2.neural.resnet34mtl.2024.12.29.12.41.Fractal-Define-XL-R2" -p "82.awa2.pc.pgd.2024.12.29.12.41.Fractal-Define-XL-R2"
./test_npc.py -r "42.celeba.neural.resnet34mtl.2024.12.18.23.46.PowerEdge-R720" -p "42.celeba.pc.pgd.2024.12.18.23.46.PowerEdge-R720"
./test_npc.py -r "52.celeba.neural.resnet34mtl.2024.12.29.23.15.Aurora-R11" -p "52.celeba.pc.pgd.2024.12.29.23.15.Aurora-R11"
./test_npc.py -r "62.celeba.neural.resnet34mtl.2024.12.30.15.46.Aurora-R11" -p "62.celeba.pc.pgd.2024.12.30.15.46.Aurora-R11"
./test_npc.py -r "72.celeba.neural.resnet34mtl.2024.12.29.23.22.Fractal-Define-R5-TV" -p "72.celeba.pc.pgd.2024.12.29.23.22.Fractal-Define-R5-TV"
./test_npc.py -r "82.celeba.neural.resnet34mtl.2024.12.30.21.28.Fractal-Define-R5-TV" -p "82.celeba.pc.pgd.2024.12.30.21.28.Fractal-Define-R5-TV"
./test_npc.py -r "42.gtsrb.neural.resnet34mtl.2024.12.13.16.47.PowerEdge-R720" -p "42.gtsrb.pc.pgd.2024.12.13.16.47.PowerEdge-R720"
./test_npc.py -r "52.gtsrb.neural.resnet34mtl.2024.12.29.0.50.Fractal-Define-R5-TV" -p "52.gtsrb.pc.pgd.2024.12.29.0.50.Fractal-Define-R5-TV"
./test_npc.py -r "62.gtsrb.neural.resnet34mtl.2024.12.29.4.49.Fractal-Define-R5-TV" -p "62.gtsrb.pc.pgd.2024.12.29.4.49.Fractal-Define-R5-TV"
./test_npc.py -r "72.gtsrb.neural.resnet34mtl.2024.12.29.8.47.Fractal-Define-R5-TV" -p "72.gtsrb.pc.pgd.2024.12.29.8.47.Fractal-Define-R5-TV"
./test_npc.py -r "82.gtsrb.neural.resnet34mtl.2024.12.29.12.44.Fractal-Define-R5-TV" -p "82.gtsrb.pc.pgd.2024.12.29.12.44.Fractal-Define-R5-TV"
./test_npc.py -r "42.mnist.neural.resnet34mtl.2024.12.13.16.52.PowerEdge-R720" -p "42.mnist.pc.pgd.2024.12.13.16.52.PowerEdge-R720"
./test_npc.py -r "52.mnist.neural.resnet34mtl.2024.12.29.0.48.Aurora-R11" -p "52.mnist.pc.pgd.2024.12.29.0.48.Aurora-R11"
./test_npc.py -r "62.mnist.neural.resnet34mtl.2024.12.29.3.10.Aurora-R11" -p "62.mnist.pc.pgd.2024.12.29.3.10.Aurora-R11"
./test_npc.py -r "72.mnist.neural.resnet34mtl.2024.12.29.5.33.Aurora-R11" -p "72.mnist.pc.pgd.2024.12.29.5.33.Aurora-R11"
./test_npc.py -r "82.mnist.neural.resnet34mtl.2024.12.29.7.57.Aurora-R11" -p "82.mnist.pc.pgd.2024.12.29.7.57.Aurora-R11"

# Jointly optimized with knowledge-injected PC
./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.13.22.27.PowerEdge-R720" -p "42.awa2.pc.pgd.2024.12.13.22.27.PowerEdge-R720"
./test_npc.py -r "52.awa2.neural.resnet34mtl.2024.12.29.0.46.Fractal-Define-XL-R2" -p "52.awa2.pc.pgd.2024.12.29.0.46.Fractal-Define-XL-R2"
./test_npc.py -r "62.awa2.neural.resnet34mtl.2024.12.29.5.6.Fractal-Define-XL-R2" -p "62.awa2.pc.pgd.2024.12.29.5.6.Fractal-Define-XL-R2"
./test_npc.py -r "72.awa2.neural.resnet34mtl.2024.12.29.9.31.Fractal-Define-XL-R2" -p "72.awa2.pc.pgd.2024.12.29.9.31.Fractal-Define-XL-R2"
./test_npc.py -r "82.awa2.neural.resnet34mtl.2024.12.29.13.51.Fractal-Define-XL-R2" -p "82.awa2.pc.pgd.2024.12.29.13.51.Fractal-Define-XL-R2"
./test_npc.py -r "42.celeba.neural.resnet34mtl.2024.12.20.17.17.PowerEdge-R720" -p "42.celeba.pc.pgd.2024.12.20.17.17.PowerEdge-R720"
./test_npc.py -r "52.celeba.neural.resnet34mtl.2024.12.29.23.8.Fractal-Define-XL-R2" -p "52.celeba.pc.pgd.2024.12.29.23.8.Fractal-Define-XL-R2"
./test_npc.py -r "62.celeba.neural.resnet34mtl.2024.12.30.18.55.Fractal-Define-XL-R2" -p "62.celeba.pc.pgd.2024.12.30.18.55.Fractal-Define-XL-R2"
./test_npc.py -r "72.celeba.neural.resnet34mtl.2024.12.29.23.9.Fractal-Define-XL-R2" -p "72.celeba.pc.pgd.2024.12.29.23.9.Fractal-Define-XL-R2"
./test_npc.py -r "82.celeba.neural.resnet34mtl.2024.12.30.19.1.Fractal-Define-XL-R2" -p "82.celeba.pc.pgd.2024.12.30.19.1.Fractal-Define-XL-R2"
./test_npc.py -r "42.gtsrb.neural.resnet34mtl.2024.12.13.23.55.PowerEdge-R720" -p "42.gtsrb.pc.pgd.2024.12.13.23.55.PowerEdge-R720"
./test_npc.py -r "52.gtsrb.neural.resnet34mtl.2024.12.29.0.53.PowerEdge-R720" -p "52.gtsrb.pc.pgd.2024.12.29.0.53.PowerEdge-R720"
./test_npc.py -r "62.gtsrb.neural.resnet34mtl.2024.12.29.7.34.PowerEdge-R720" -p "62.gtsrb.pc.pgd.2024.12.29.7.34.PowerEdge-R720"
./test_npc.py -r "72.gtsrb.neural.resnet34mtl.2024.12.29.14.20.PowerEdge-R720" -p "72.gtsrb.pc.pgd.2024.12.29.14.20.PowerEdge-R720"
./test_npc.py -r "82.gtsrb.neural.resnet34mtl.2024.12.29.21.8.PowerEdge-R720" -p "82.gtsrb.pc.pgd.2024.12.29.21.8.PowerEdge-R720"
./test_npc.py -r "42.mnist.neural.resnet34mtl.2024.12.14.8.8.PowerEdge-R720" -p "42.mnist.pc.pgd.2024.12.14.8.8.PowerEdge-R720"
./test_npc.py -r "52.mnist.neural.resnet34mtl.2024.12.29.0.49.PowerEdge-R720" -p "52.mnist.pc.pgd.2024.12.29.0.49.PowerEdge-R720"
./test_npc.py -r "62.mnist.neural.resnet34mtl.2024.12.29.6.8.PowerEdge-R720" -p "62.mnist.pc.pgd.2024.12.29.6.8.PowerEdge-R720"
./test_npc.py -r "72.mnist.neural.resnet34mtl.2024.12.29.11.27.PowerEdge-R720" -p "72.mnist.pc.pgd.2024.12.29.11.27.PowerEdge-R720"
./test_npc.py -r "82.mnist.neural.resnet34mtl.2024.12.29.16.46.PowerEdge-R720" -p "82.mnist.pc.pgd.2024.12.29.16.46.PowerEdge-R720"
