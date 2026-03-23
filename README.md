## PrvTel

This repository contains the code for the PrvTel project, which aims to generate synthetic telemetry data with differential privacy.

## Getting Started

### Installation

Before proceeding, make sure you have working Conda installation.

To create a suitable environment:
- `cd path/to/vae_cloud_computing`
- `conda env create --file prvtel.yml`
- `conda activate prvtel`

### GPU Support

This code has been tested both on CPU in the torch v1.9.0 given. But it has also been run on a GPU environment. The specifications for the device running this are as follows:

- NVIDIA GeForce RTX 4090
- CUDA v12.6
- cuDNN v9.3.0.75

## Usage

To train a VAE model on Appraise data, first train the word2vec embeddings and apply them to the data. Make sure that Appraise data is available to train the word2vec model. Then, run the provided training and inference scripts.
```bash
# Train word2vec embeddings for ports and protocols on Appraisea.
cd PrvTel/scripts/port2vec
./create_embeddings.sh # Modify script to transform correct dataset.

# Run train script and inference script.
cd PrvTel
./experiments/appraise_train.sh 
./experiments/appraise_inf.sh
```
Training for CAIDA and NF-IoT datasets follow a similar procedure with their corresponding scripts. Scripts used to compute statistics can be found in the `scripts/stats` folder. A more detailed overview of the project structure can be found below:

```
PrvTel/
├── ...
├── prvtel /
│   └── # Contains main source code.
├── experiments/
│   └── # Scripts for training and syn data gen
├── scripts/
│   ├── port2vec/
│   │   └── # Training word2vec and transform/inverse transform code.
│   ├── privbayes/
│   │   └── # Code for injecting PrivBayes noise.
│   ├── sketches/
│   │   └── # Benchmarking various sketch algorithms.
│   ├── stats/
│   │   └── # Computing various evaluation statistics for synthetic data.
│   └── classifier_eval.ipynb # Evaluating effect of PrivBayes noise on ML classification.
├── dist_train.py
├── generate.py
└── ...
```

## Datasets

The following datasets were used in this project:

- **[Appraise H2020](https://www.kaggle.com/datasets/ittibydgoszcz/appraise-h2020-real-labelled-netflow-dataset?resource=download)**
- **[NF-UQ-NIDS-v2 (NF IoT)](https://espace.library.uq.edu.au/view/UQ:631a24a)**
- **[CAIDA Anonymized Internet Traces](https://www.caida.org/catalog/datasets/passive_dataset/)**
- **[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)**
- **[NERC](https://nerc.mghpcc.org/project/open-cloud-testbed/)**
- **[Cisco IE](https://github.com/cisco-ie/telemetry)**

For Cisco IE we also provide helper script for downloading the specific data we used. More details on each dataset and how they were processed can be found in the paper. See Table 3 for a quick summary.




