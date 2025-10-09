# Neural Probabilistic Circuit

## Table of Contents

1. [Training](#training)
1. [Testing](#testing)
1. [Pretrained Checkpoint Files](#pretrained-checkpoint-files)

## Training

Before starting, review and adjust additional parameters, such as training hyperparameters, if applicable, in `config_neural` and `config_pc` within `header.py`.

At the start of each training, a unique run name is automatically generated. During training, two checkpoint files containing model weights are produced, one updated every epoch and another updated only when validation performance improves, i.e., the _best_ checkpoint.

Checkpoint files are stored under `npc-models/outputs/npc-models/checkpoints` and named using the generated run name. The best checkpoint file ends with `.best.zip`, as defined in `header.py`, and should be used for joint optimization and testing.

If Weights & Biases (wandb) is enabled, a wandb run is automatically created under the generated run name, and all checkpoints are uploaded to the wandb cloud storage.

Before joint optimization and testing, ensure that the dataset prefix in `header.py` matches the one specified in the run name or checkpoint file name passed as arguments.

For joint optimization and testing involving PCs constructed using the knowledge-injected approach, modify the following parameter in `header.py` to point to the knowledge-injected PC model constructed by the `learnspn` project:

```bash
config_pc = {
    ...
    "file_path_pc": "../../../learnspn/outputs/manual/" + dataset_prefix + ".spn.txt",
    ...
}
```

Be sure to revert this parameter when switching back to the data-driven approach.

Refer to the training bash scripts in `npc/npc-models/script` for examples on batch training and the use of pretrained checkpoint files from the paper experiments.

### Stage 1: Neural Attribute Recognition

Train the Neural Attribute Recognition model:

```bash
cd npc/npc-models/src/npc-models
./train_neural.py -b 256 -e 150 -s 42
```

Arguments:

- `-b`: Batch size.
- `-e`: Epochs.
- `-s`: Seed.

### Stage 2: Probabilistic Circuit (PC)

Train the PC model constructed using the data-driven approach via CCCP parameter learning:

```bash
cd npc/npc-models/src/npc-models
./train_pc.py -e 50 -s 42
```

Arguments:

- `-e`: Epochs.
- `-s`: Seed.

PC models constructed using the knowledge-injected approach do not require training in Stage 2, as the approach does not involve parameter learning.

### Stage 3: Joint Optimization

For the data-driven approach, jointly optimize the independently trained Neural and PC models:

```bash
cd npc/npc-models/src/npc-models
./train_npc.py -w "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11.best.zip" -c "42.awa2.pc.cccp.2024.12.12.19.55.Aurora-R11.best.zip" -b 256 -e 150 -s 42
```

Arguments:

- `-w`: Neural model pretrained weights (full checkpoint file name, not run name).
- `-c`: PC model pretrained weights (full checkpoint file name, not run name).
- `-b`: Batch size.
- `-e`: Epochs.
- `-s`: Seed.

Use `-w` to specify the best Neural model checkpoint from Stage 1 and `-c` for the best PC model checkpoint from Stage 2.

For the knowledge-injected approach, use the same command but omit `-c`.

## Testing

At the end of each training session, the training scripts automatically test the trained models. The identical test can be repeated manually by running the testing scripts and passing the training run names as arguments. Note that wandb is automatically disabled in the testing scripts.

Refer to the testing bash scripts in `npc/npc-models/script` for examples on batch training and the use of pretrained checkpoint files from the paper experiments.

### Neural Attribute Recognition

Test the trained Neural Attribute Recognition model:

```bash
cd npc/npc-models/src/npc-models
./test_neural.py -r "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11"
```

Arguments:

- `-r`: Run name.

### Probabilistic Circuit (PC)

Test the trained PC model:

```bash
cd npc/npc-models/src/npc-models
./test_pc.py -p "42.awa2.pc.cccp.2024.12.12.19.55.Aurora-R11"
```

Arguments:

- `-p`: Run name.

### Neural Probabilistic Circuit (NPC)

Test the trained Neural and PC models, either independently trained or jointly optimized, as a complete NPC pipeline:

```bash
cd npc/npc-models/src/npc-models
./test_npc.py -r "42.awa2.neural.resnet34mtl.2024.12.12.20.5.Aurora-R11" -p "42.awa2.pc.cccp.2024.12.12.19.55.Aurora-R11"
```

Arguments:

- `-r`: Neural run name.
- `-p`: PC run name.

## Pretrained Checkpoint Files

Pretrained checkpoint files from the paper experiments are not released by default. Contact [Simon Yu](mailto:simonyu@simonyu.net) to request access. All pretrained checkpoint files should be placed under `npc-models/outputs/npc-models/checkpoints`. Create the directories if they do not already exist.

Written by [Simon Yu](https://www.simonyu.net/).
