# Inference

We provide information and example scripts on how to perform inference using EgoM2P models.

## Structure

#### Important files and directories:

#### `egom2p/models`
- `egom2p/models/generate.py`: Contains most of the generation logic for different generation schedules.

#### `egom2p/utils/`
- `utils/generation.py`: Contains helper functions for building generation schedules.
- `utils/plotting_utils.py`: Contains helper functions for decoding and plotting different modalities.


#### Scripts and notebooks:
- `eval_model_*2*.py`: Script that perform the benchmarked four tasks in EgoM2P.


## Generation parameters

`utils/generation.py` contains a function to build a _generation schedule_ to specify which modality to generate with which settings, as well as a Sampler that wraps a trained EgoM2P model and performs the generation using various generation schemes when provided an input and generation schedule.

Inputs to the model are given as dictionaries of modalities, where each modality in turn is a dictionary that specifies a set of tokens that can contain data to input to the model or placeholder values to fill in during generation, an input mask that specifies which parts of the given tokens are used as input to the model, and a target mask that specifies which parts of the tokens are to be predicted. 

#### Condition & target domains
- `cond_domains`: Domains that are used as conditioning.
- `target_domains`: Domains that are predicted. The order given is the order in which the domains are predicted one-by-one. We did not provide chained generation in the eval scripts, e.g., rgb -> depth -> cam. But the model supports such inference generation schedule.

#### Generation settings
- `tokens_per_target`: Number of tokens to decode per target modality. 
- `autoregression_schemes`: Generation scheme for each target modality. `maskgit` or `roar` for image-like modalities, and `autoregressive` for sequence modalities. We treat videos and motion like a big image, and use `roar` as the scheme.
- `decoding_steps`: Integers that specify in how many steps each target modality should be decoded. For example, if predicting a video of 5120 tokens in 10 steps with a linear decoding schedule, each step decodes 512 tokens, or when specifying 1 step, the entire video is predicted in a single forward pass. 
- `token_decoding_schedules`: Type of decoding schedule that specifies how many tokens are being decoded at which decoding step (if applicable). `cosine` starts and ends with a small number of decoded tokens, but decodes many in the middle of the schedule. `linear` decodes the same number of tokens each time step.

#### Temperature settings
- `temps`: Sampling temperatures for each target modality.
- `temp_schedules`: Temperature sampling schedules for each target modality. `constant` keeps the temperature constant for the duration of decoding, `linear` linearly decays the temperature from the indicated temperature down to zero, and `onex:{min_t}:{power}` decays the temperature proportional to x^-power from the starting temperature until the minimum temperature min_t.

#### Classifier-free guidance settings
- `cfg_scales`: Classifier-free guidance scales for each target modality. A value of 1.0 means no guidance is performed. Values > 1.0 perform positive guidance, values between 0.0 and 1.0 perform weak guidance, 0.0 is equal to an unconditional case, and lower values perform negative guidance.
- `cfg_schedules`: Only the `constant` schedule is implemented at the moment.
- `cfg_grow_conditioning`: True or False. If True, each completed modality is added to the classifier-free guidance conditioning.

#### Top-k & top-p sampling settings
- `top_p`: When top_p > 0.0, keep only the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering).
- `top_k`: When top_k > 0, keep only the top k tokens with highest probability (a.k.a. top-k filtering).

Performing generation with EgoM2P can be complex due to the large range of possibilities that come with all the ways chained generation can be performed, the different available generation schedules and their hyperparameters, etc. Feel free to experiment!
