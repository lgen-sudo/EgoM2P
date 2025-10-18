# EgoM2P Training

We provide instructions on how to pretrain EgoM2P models and where to find the most relevant files. Please see the EgoM2P paper for more background on our pretraining strategy. For instructions on how to train prepare the training datasets and train the tokenizers, please see [README_DATA.md](README_DATA.md) and [README_TOKENIZATION.md](README_TOKENIZATION.md).

## Structure

#### `egom2p/models/`
- Code related to the EgoM2P models
- Some important files:
    - `models/egom2p_model.py`: Main file defining the `EgoM2P` module, containing all model architecture definitions and forward pass logic.
    - `models/encoder_embeddings.py` and `models/decoder_embeddings.py`: Contains per-modality modules that map the input tokens to embeddings, and the embeddings to logits. Also adds the positional and modality embeddings to tokens.
    - `generate.py`: Contains sampling logic & utilities for any-to-any generation with trained EgoM2P models.

#### `egom2p/data/`
- Handles data loading, preparation, and input/target masking.
- Some important files:
    - `data/modality_info.py`: Defines modality metadata like name, type, vocabulary size, and encoder/decoder embedding modules. In EgoM2P, we treat all modalities (video and motion) as a "big image", and use image-like masking strategies. 
    - `data/unified_datasets.py`: Loads aligned multimodal datasets, either locally or from cloud object stores (e.g. S3).
    - `data/masking.py`: Performs multimodal input/target masking based on provided token budgets and Dirichlet sampling parameters.


## General information

### Configs

Training runs are configured using YAML files that are organized as follows:
- Main training config: Contains most training information and hyperparameters (e.g. model architecture details, number of training steps, etc.), as well as logging and saving information. See [here](cfgs/default/egom2p/models/main/ego-b_mod4_500b_clariden_2048_camcv_depthdenoise.yaml) for an example.
- Data config: Provides details about the training data mix, including source datasets, input and target modalities, dataset paths, modality name mappings, etc. See [here](cfgs/default/egom2p/data/ego/main/mix_mod4_all2all_2048.yaml) for an example. The path to the data config needs to be specified in the main training config.
- Alphas configs: Defines the Dirichlet distribution hyperparameters used to sample proportions of tokens from each modality during training, and enables defining mixture of Dirichlet distributions. See [here](cfgs/default/egom2p/alphas_mixture/main/mix_mod4_all2all_uni.yaml) for an example. The path(s) to the alphas config(s) need to be specified in the data config.

Optionally, command-line arguments can be used to override some config information. To modify training settings, either either edit / add config files or provide additional command-line arguments.


### Training EgoM2P Models

The EgoM2P training script supports multi-node training with PyTorch Distributed Data Parallel (DDP).

To train a EgoM2P model using DDP (recommended for B-sized models) on a 4 GPU node, run:

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 run_training_egom2p.py \
--config cfgs/default/egom2p/models/<config>.yaml
```

The training slurm script for the released checkpoint is [here](train_slurm_script/clariden_main_500b_2048.slurm). We train the model with 256 H100 GPUs for 15 hours. For larger model size / longer training length, we need scale dataset size in egocentric vision.
