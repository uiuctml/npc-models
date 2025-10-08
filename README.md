# Neural Probabilistic Circuit Models

## Table of Contents

1. [Project Overview](#project-overview)
1. [Project Prerequisites](#project-prerequisites)
1. [Project Hierarchy](#project-hierarchy)
1. [Publications](#publications)
1. [Acknowledgements](#acknowledgements)
1. [License](#license)
1. [Contact](#contact)

## Project Overview



## Project Prerequisites

This project requires the following system packages:

Ubuntu:

```bash
apt install libgl1-mesa-dev python3-venv
```

Arch Linux:

```bash
pacman -S mesa
```

This project was developed on Ubuntu and tested on both Ubuntu and Arch Linux. Other Linux distributions, macOS, or Windows Subsystem for Linux (WSL) may also work with additional setup. However, these platforms are not officially supported.

To ensure maximum compatibility and performance, run this project on a system with at least 64 GB of CPU memory (swap space acceptable) and 16 GB of GPU memory (aggregate across all available GPUs).

## Project Hierarchy

This project is part of the NPC pipeline. To ensure compatibility and maintain consistent references across the pipeline, organize the project directories as follows:

    npc
    ├── datasets
    ├── learnspn
    ├── npc-dataset-utils
    ├── npc-models
    └── npc-venv

All subsequent instructions assume the above project hierarchy.

This project is designed to run within a dedicated Python virtual environment. Create and activate the environment as follows:

```bash
cd npc
deactivate
python3 -m venv npc-venv
source npc-venv/bin/activate
python3 -m pip install -r npc-dataset-utils/requirements.txt
```

Always ensure the virtual environment is activated before running the project.

Before running this project, first ensure that all datasets are properly set up under `npc/datasets` by following the instructions in the `npc-dataset-utils` project. Then, construct and generate PCs for all datasets as described in the `learnspn` project instructions.

## Publications

Upon using this project, cite any relevant publications listed below:

### Neural Probabilistic Circuit (NPC)

```
@article{chen2025neural,
  title={Neural probabilistic circuits: Enabling compositional and interpretable predictions through logical reasoning},
  author={Chen, Weixin and Yu, Simon and Shao, Huajie and Sha, Lui and Zhao, Han},
  journal={arXiv preprint arXiv:2501.07021},
  year={2025}
}
```

```
@inproceedings{chenneural,
  title={Neural Probabilistic Circuits: An Overview},
  author={Chen, Weixin and Yu, Simon and Shao, Huajie and Sha, Lui and Zhao, Han},
  booktitle={Eighth Workshop on Tractable Probabilistic Modeling}
}
```

### Probabilistic Circuit (PC)

```
@article{zhao2016unified,
  title={A unified approach for learning the parameters of sum-product networks},
  author={Zhao, Han and Poupart, Pascal and Gordon, Geoffrey J},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}
```

### Residual Network (ResNet)

```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

### Concept Bottleneck Model (CBM)

```
@inproceedings{koh2020concept,
  title={Concept bottleneck models},
  author={Koh, Pang Wei and Nguyen, Thao and Tang, Yew Siang and Mussmann, Stephen and Pierson, Emma and Kim, Been and Liang, Percy},
  booktitle={International conference on machine learning},
  pages={5338--5348},
  year={2020},
  organization={PMLR}
}
```

### Concept Embedding Model (CEM)

```
@article{espinosa2022concept,
  title={Concept embedding models: Beyond the accuracy-explainability trade-off},
  author={Espinosa Zarlenga, Mateo and Barbiero, Pietro and Ciravegna, Gabriele and Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Shams, Zohreh and Precioso, Frederic and Melacci, Stefano and Weller, Adrian and others},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={21400--21413},
  year={2022}
}
```

### Deep Concept Reasoner (DCR)

```
@inproceedings{barbiero2023interpretable,
  title={Interpretable neural-symbolic concept reasoning},
  author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Zarlenga, Mateo Espinosa and Magister, Lucie Charlotte and Tonda, Alberto and Li{\'o}, Pietro and Precioso, Frederic and Jamnik, Mateja and Marra, Giuseppe},
  booktitle={International Conference on Machine Learning},
  pages={1801--1825},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgements

Special thanks to Rahim Khan, Tommy Tang, Alex Tanthiptham, and Trusha Vernekar for their contributions to the implementation, testing, and experiments involved in this project.

## License

This codebase is released under the [Creative Commons Attribution NonCommercial ShareAlike (CC BY-NC-SA)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) license, which can be viewed under `LICENSE`.

## Contact

For questions, feedback, or comments, open an issue or reach out to [Simon Yu](mailto:simonyu@simonyu.net).

Written by [Simon Yu](https://www.simonyu.net/).
