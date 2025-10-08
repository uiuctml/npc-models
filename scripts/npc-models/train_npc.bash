#!/usr/bin/env bash

cd "../../src/npc-models"

# Jointly optimized with data-driven PC
./train_npc.py -w "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11.best.zip" -c "42.awa2.pc.cccp.2024.12.12.19.55.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.awa2.neural.resnet34mtl.2024.12.24.0.30.Aurora-R11.best.zip" -c "52.awa2.pc.cccp.2024.12.22.15.16.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.awa2.neural.resnet34mtl.2024.12.24.3.25.Aurora-R11.best.zip" -c "62.awa2.pc.cccp.2024.12.22.15.17.Aurora-R11.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.awa2.neural.resnet34mtl.2024.12.24.6.21.Aurora-R11.best.zip" -c "72.awa2.pc.cccp.2024.12.22.15.17.Aurora-R11.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.awa2.neural.resnet34mtl.2024.12.24.9.16.Aurora-R11.best.zip" -c "82.awa2.pc.cccp.2024.12.22.15.18.Aurora-R11.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.celeba.neural.resnet34mtl.2024.12.17.21.56.Aurora-R11.best.zip" -c "42.celeba.pc.cccp.2024.12.17.21.53.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.celeba.neural.resnet34mtl.2024.12.26.2.57.Fractal-Define-XL-R2.best.zip" -c "52.celeba.pc.cccp.2024.12.22.15.24.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.celeba.neural.resnet34mtl.2024.12.26.19.16.Fractal-Define-XL-R2.best.zip" -c "62.celeba.pc.cccp.2024.12.22.15.26.Aurora-R11.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.celeba.neural.resnet34mtl.2024.12.26.2.58.Fractal-Define-XL-R2.best.zip" -c "72.celeba.pc.cccp.2024.12.22.15.28.Aurora-R11.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.celeba.neural.resnet34mtl.2024.12.26.19.20.Fractal-Define-XL-R2.best.zip" -c "82.celeba.pc.cccp.2024.12.22.15.29.Aurora-R11.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.gtsrb.neural.resnet34mtl.2024.12.10.18.10.PowerEdge-R720.best.zip" -c "42.gtsrb.pc.cccp.2024.12.11.16.22.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.gtsrb.neural.resnet34mtl.2024.12.24.17.41.Aurora-R11.best.zip" -c "52.gtsrb.pc.cccp.2024.12.22.15.33.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.gtsrb.neural.resnet34mtl.2024.12.24.5.54.PowerEdge-R720.best.zip" -c "62.gtsrb.pc.cccp.2024.12.22.15.33.Aurora-R11.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.gtsrb.neural.resnet34mtl.2024.12.24.11.37.PowerEdge-R720.best.zip" -c "72.gtsrb.pc.cccp.2024.12.22.15.34.Aurora-R11.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.gtsrb.neural.resnet34mtl.2024.12.24.17.21.PowerEdge-R720.best.zip" -c "82.gtsrb.pc.cccp.2024.12.22.15.34.Aurora-R11.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.mnist.neural.resnet34mtl.2024.12.10.18.11.PowerEdge-R720.best.zip" -c "42.mnist.pc.cccp.2024.12.11.16.23.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.mnist.neural.resnet34mtl.2024.12.24.0.10.PowerEdge-R720.best.zip" -c "52.mnist.pc.cccp.2024.12.22.15.53.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.mnist.neural.resnet34mtl.2024.12.24.4.40.PowerEdge-R720.best.zip" -c "62.mnist.pc.cccp.2024.12.22.15.54.Aurora-R11.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.mnist.neural.resnet34mtl.2024.12.24.9.9.PowerEdge-R720.best.zip" -c "72.mnist.pc.cccp.2024.12.22.15.55.Aurora-R11.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.mnist.neural.resnet34mtl.2024.12.24.13.37.PowerEdge-R720.best.zip" -c "82.mnist.pc.cccp.2024.12.22.15.56.Aurora-R11.best.zip" -b 256 -e 150 -s 82

# Jointly optimized with knowledge-injected PC
./train_npc.py -w "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.awa2.neural.resnet34mtl.2024.12.24.0.30.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.awa2.neural.resnet34mtl.2024.12.24.3.25.Aurora-R11.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.awa2.neural.resnet34mtl.2024.12.24.6.21.Aurora-R11.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.awa2.neural.resnet34mtl.2024.12.24.9.16.Aurora-R11.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.celeba.neural.resnet34mtl.2024.12.17.21.56.Aurora-R11.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.celeba.neural.resnet34mtl.2024.12.26.2.57.Fractal-Define-XL-R2.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.celeba.neural.resnet34mtl.2024.12.26.19.16.Fractal-Define-XL-R2.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.celeba.neural.resnet34mtl.2024.12.26.2.58.Fractal-Define-XL-R2.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.celeba.neural.resnet34mtl.2024.12.26.19.20.Fractal-Define-XL-R2.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.gtsrb.neural.resnet34mtl.2024.12.10.18.10.PowerEdge-R720.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.gtsrb.neural.resnet34mtl.2024.12.24.17.41.Aurora-R11.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.gtsrb.neural.resnet34mtl.2024.12.24.5.54.PowerEdge-R720.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.gtsrb.neural.resnet34mtl.2024.12.24.11.37.PowerEdge-R720.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.gtsrb.neural.resnet34mtl.2024.12.24.17.21.PowerEdge-R720.best.zip" -b 256 -e 150 -s 82
./train_npc.py -w "42.mnist.neural.resnet34mtl.2024.12.10.18.11.PowerEdge-R720.best.zip" -b 256 -e 150 -s 42
./train_npc.py -w "52.mnist.neural.resnet34mtl.2024.12.24.0.10.PowerEdge-R720.best.zip" -b 256 -e 150 -s 52
./train_npc.py -w "62.mnist.neural.resnet34mtl.2024.12.24.4.40.PowerEdge-R720.best.zip" -b 256 -e 150 -s 62
./train_npc.py -w "72.mnist.neural.resnet34mtl.2024.12.24.9.9.PowerEdge-R720.best.zip" -b 256 -e 150 -s 72
./train_npc.py -w "82.mnist.neural.resnet34mtl.2024.12.24.13.37.PowerEdge-R720.best.zip" -b 256 -e 150 -s 82
