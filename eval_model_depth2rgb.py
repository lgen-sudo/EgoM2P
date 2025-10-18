import numpy as np
import torch
import glob
import os
from run_training_egom2p import get_model, setup_data
from egom2p.models.generate import GenerationSampler, build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality
from egom2p.data.modality_info import MODALITY_INFO
from egom2p.utils.plotting_utils import decode_rgb
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import time
from eval_model_rgb2depth import encode_mp4_to_token

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

vid_tok = CausalVideoTokenizer(checkpoint_dec=f'./ckpt/decoder.jit') # download Cosmos-0.1-Tokenizer-DV4x8x8 decoder.jit to this folder

toks = {
    'tok_vid': vid_tok,
}

checkpoint = torch.load('./ckpt/checkpoint-main.pth', map_location='cpu', weights_only=False)
args = checkpoint['args']
model = get_model(args, setup_data(args)[0])
model.load_state_dict(checkpoint['model'])
model = model.eval().to(device)
sampler = GenerationSampler(model)

cond_domains = ['tok_depth']
target_domains = ['tok_rgb']
tokens_per_target = [5120]
autoregression_schemes = ['roar'] * len(target_domains)

# NOTE: you can tune this parameter for best eval performance
decoding_steps = [6] * len(target_domains)
token_decoding_schedules = ['linear'] * len(target_domains)
temps = [0.01] * len(target_domains)

temp_schedules = ['constant'] * len(target_domains)
cfg_scales = [2.0] * len(target_domains)
cfg_schedules = ['constant'] * len(target_domains)
cfg_grow_conditioning = True
top_p, top_k = 0.8, 0.0

schedule = build_chained_generation_schedules(
    cond_domains=cond_domains, target_domains=target_domains, tokens_per_target=tokens_per_target, autoregression_schemes=autoregression_schemes, 
    decoding_steps=decoding_steps, token_decoding_schedules=token_decoding_schedules, temps=temps, temp_schedules=temp_schedules,
    cfg_scales=cfg_scales, cfg_schedules=cfg_schedules, cfg_grow_conditioning=cfg_grow_conditioning, 
)

for file in glob.glob('example_data/dep2rgb_*.mp4'): # make sure all videos are in 8fps, 256x256, 2-second
    name = os.path.basename(file)
    print(name)

    # convert mp4 to tokens
    input_token = encode_mp4_to_token(file)[0]

    start = time.time()

    batched_sample = {
        'tok_depth': {
            'tensor': torch.as_tensor(input_token).unsqueeze(0).to(device), # Batched tensor
            'input_mask': torch.zeros(1, 5120, dtype=torch.bool, device=device), # False = used as input, True = ignored
            'target_mask': torch.ones(1, 5120, dtype=torch.bool, device=device), # False = predicted as target, True = ignored
        }
    }
    
    # Initialize target modalities
    for target_mod, ntoks in zip(target_domains, tokens_per_target):
        batched_sample = init_empty_target_modality(batched_sample, MODALITY_INFO, target_mod, 1, ntoks, device)
    
    # Initialize input modalities
    for cond_mod in cond_domains:
        batched_sample = init_full_input_modality(batched_sample, MODALITY_INFO, cond_mod, device)

    out_dict = sampler.generate(
        batched_sample, schedule, 
        verbose=False, seed=0,
        top_p=top_p, top_k=top_k,
    )
    
    dec_dict = decode_rgb(
        name,
        out_dict, toks,
        image_size=256, patch_size=8,
        name = 'example_data/dep2rgb'
    )

    print(f'predicting {file}')