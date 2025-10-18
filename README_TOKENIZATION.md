# Tokenization
We use tokenization to convert diverse multi-modalities (RGB video, depth video, gaze dynamics and camera trajectory) into a unified representation space. For video modalities, we use recent [Cosmos tokenizer](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-DV4x8x8) to tokenize them into discrete tokens. We train our own gaze and camera pose tokenizer.

## Structure

#### `egom2p/vq/`
- All code related to the tokenizers (vq = **v**ector **q**uantization)
- Some important files and directories:
    - `vq/models/`: Contains all the encoder and decoder architectures.
    - `vq/quantizers/`: Contains different quantizer implementations.
    - `vq/vqvae.py`: Main file defining the standard VQ-VAE classes.


#### Root directory
- `run_training_vqvae.py`: Main training script for training gaze dynamics and camera pose VQ-VAEs.


## VQ-VAE

### Tokenizer inference

* Tokenization with example data: See [cam_tok](example_data/tok_cam_example.py) and [gaze_tok](example_data/tok_gaze_example.py).

```
python example_data/tok_cam_example.py 
python example_data/tok_gaze_example.py 
```

* See how to tokenize datasets into tokens: [camera script](tokenize_script/cam.sh) and [gaze script](tokenize_script/gaze.sh).

### Tokenizer training

First you need to prepare and convert raw data into a npy training/val file as [load_cam.py](egom2p/data/load_cam.py) and [load_gaze.py](egom2p/data/load_cam.py). The camera trajectory should be in opencv format, cam2world convention. Gaze dynamics should also be in opencv convention, i.e., x is the horizontal coordinate and y is the vertical coordinate.

VQ-VAE training configs can be found in [cfgs/default/tokenization/vqvae](cfgs/default/tokenization/vqvae). To train a VQ-VAE on a 4 GPU node, run:
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 run_training_vqvae.py \
--config cfgs/default/tokenization/vqvae/<config>.yaml
```

The camera tokenizer training slurm script is [here](train_slurm_script/cam_tok_train.slurm). The gaze tokenizer training slurm script is [here](train_slurm_script/gaze_train.slurm).

## Pre-computing tokens for efficient EgoM2P training

Once the tokenizer networks are trained, please follow the instructions in [Tokenizer inference](#tokenizer-inference) on how to pre-compute the tokens. 
