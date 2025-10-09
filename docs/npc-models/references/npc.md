# Command Reference for NPC Models

## Table of Contents

1. [Training](#training)
1. [Testing](#testing)

## Training

For joint optimization and testing involving PCs constructed using the knowledge-injected approach, modify the following parameter in `header.py` to point to the knowledge-injected PC model constructed by the `learnspn` project:

```python
config_pc = {
    ...
    "file_path_pc": "../../../learnspn/outputs/manual/" + dataset_prefix + ".spn.txt",
    ...
}
```

Be sure to revert this parameter when switching back to the data-driven approach.

### Stage 1: Neural Attribute Recognition

Train the Neural Attribute Recognition model:

```bash
cd npc/npc-models/src/npc-models
./train_neural.py -b <batches> -e <epochs> -s <seed>
```

Arguments:

- `-b`: Batch size.
- `-e`: Epochs.
- `-s`: Seed.

### Stage 2: Probabilistic Circuit (PC)

Train the PC model constructed using the data-driven approach via [CCCP](https://proceedings.neurips.cc/paper/2016/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) parameter learning:

```bash
cd npc/npc-models/src/npc-models
./train_pc.py -e <epochs> -s <seed>
```

Arguments:

- `-e`: Epochs.
- `-s`: Seed.

PC models constructed using the knowledge-injected approach do not require training in Stage 2, as the approach does not involve parameter learning.

### Stage 3: Joint Optimization

For the data-driven approach, jointly optimize the independently trained Neural and PC models:

```bash
cd npc/npc-models/src/npc-models
./train_npc.py -w "<neural checkpoint>" -c "<pc checkpoint>" -b <batches> -e <epochs> -s <seed>
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

### Neural Attribute Recognition

Test the trained Neural Attribute Recognition model:

```bash
cd npc/npc-models/src/npc-models
./test_neural.py -r "<run>"
```

Arguments:

- `-r`: Run name.

### Probabilistic Circuit (PC)

Test the trained PC model:

```bash
cd npc/npc-models/src/npc-models
./test_pc.py -p "<run>"
```

Arguments:

- `-p`: Run name.

### Neural Probabilistic Circuit (NPC)

Test the trained Neural and PC models, either independently trained or jointly optimized, as a complete NPC pipeline:

```bash
cd npc/npc-models/src/npc-models
./test_npc.py -r "<neural run>" -p "<pc run>"
```

Arguments:

- `-r`: Neural run name.
- `-p`: PC run name.

In addition to standard model testing, `test_npc.py` also analyzes the impact of attribute selection during inference. This functionality helps study the role and influence of each attribute within the NPC pipeline:

- _How does the exclusion of one particular attribute during inference impact the performance of NPC on downstream tasks?_
- _How does the exclusion of different attributes vary the impact?_

To exclude an attribute during testing, set the following parameter in `header.py`:

```python
npc_attribute_exclude = "<attribute>"
```

Written by [Simon Yu](https://www.simonyu.net/).
