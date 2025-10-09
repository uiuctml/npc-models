# Neural Probabilistic Circuit

## Training

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

### Notes for Stages 1 and 2

- PC models constructed using the knowledge-injected approach do not require training in Stage 2, as the approach does not involve parameter learning.
- Review and adjust additional parameters, such as training hyperparameters, if applicable, in `config_neural` and `config_pc` within `header.py`.
- Refer to `npc/npc-models/script/train_neural.bash` and `npc/npc-models/script/train_pc.bash` for examples on headless batch training.

At the start of each training, a unique run name is automatically generated. During training, two checkpoint files containing model weights are produced, one updated every epoch and another updated only when validation performance improves, i.e., the _best_ checkpoint.

Checkpoint files are stored under `npc-models/outputs/npc-models/checkpoints` and named using the generated run name. The best checkpoint file ends with `.best.zip`, as defined in `header.py`, and should be used for joint optimization and testing.

If Weights & Biases (wandb) is enabled, a wandb run is automatically created under the generated run name, and all checkpoints are uploaded to the wandb cloud storage.

### Stage 3: Joint Optimization

For the data-driven approach, jointly optimize the independently trained Neural Attribute Recognition and PC models:

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

For the knowledge-injected approach, use the same command but omit `-c`. Instead, modify the following parameter in `header.py` to point to the PC model constructed by the `learnspn` project using the knowledge-injected approach:

```bash
config_pc = {
    ...
    "file_path_pc": "../../../learnspn/outputs/manual/" + dataset_prefix + ".spn.txt",
    ...
}
```

Be sure to revert this parameter when switching back to the data-driven approach.

Refer to `npc/npc-models/script/train_npc.bash` for examples demonstrating the use of pretrained checkpoint files from the paper experiments and headless batch training. Note that the dataset prefix in `header.py` match the one specified in the checkpoint file name.

## Testing

TODO

## Pretrained Weights

Pretrained checkpoint files from the paper experiments are not released by default. Contact [Simon Yu](mailto:simonyu@simonyu.net) to request access. All pretrained checkpoint files should be placed under `npc-models/outputs/npc-models/checkpoints`. Create the directories if they do not already exist.

Written by [Simon Yu](https://www.simonyu.net/).
