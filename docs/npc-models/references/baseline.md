# Command Reference for Baseline Models

## Table of Contents

1. [Training](#training)
1. [Testing](#testing)

## Training

### Blackbox Model

Train the blackbox Residual Network (ResNet) model:

```bash
cd npc/npc-models/src/npc-models
./train_blackbox.py -b <batches> -e <epochs> -s <seed>
```

Arguments:

- `-b`: Batch size.
- `-e`: Epochs.
- `-s`: Seed.

### Other Baseline Models

Train the remaining baseline models:

```bash
cd npc/npc-models/src/npc-models
./train_baseline.py -m "<model>" -b <batches> -e <epochs> -s <seed>
```

Arguments:

- `-m`: Baseline model.
- `-b`: Batch size.
- `-e`: Epochs.
- `-s`: Seed.

`-m` currently accepts the following baseline model types:

- "abm": [Attribute Bottleneck Model (ABM)](https://arxiv.org/abs/2501.07021)
- "cbm": [Concept Bottleneck Model (CBM)](https://proceedings.mlr.press/v119/koh20a)
- "cem": [Concept Embedding Model (CEM)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html)
- "dcr": [Deep Concept Reasoner (DCR)](https://proceedings.mlr.press/v202/barbiero23a.html)

## Testing

### Blackbox Model

Test the trained blackbox model:

```bash
cd npc/npc-models/src/npc-models
./test_blackbox.py -r "<run>"
```

Arguments:

- `-r`: Run name.

### Other Baseline Models

Test any of the trained baseline models:

```bash
cd npc/npc-models/src/npc-models
./test_baseline.py -r "<run>"
```

Arguments:

- `-r`: Run name.

Written by [Simon Yu](https://www.simonyu.net/).
