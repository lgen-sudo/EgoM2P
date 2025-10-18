import numpy as np
import torch
import glob
import os
from run_training_egom2p import get_model, setup_data
from run_training_vqvae import get_model as get_tok_model
from egom2p.vq.vqvae import VQVAE
from egom2p.models.generate import GenerationSampler, build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality
from egom2p.data.modality_info import MODALITY_INFO
from egom2p.utils.plotting_utils import decode_cam
import time

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

ckpt = torch.load('./ckpt/checkpoint-cam.pth', map_location='cpu', weights_only=False)
cam_tok = get_tok_model(ckpt['args'], device)
cam_tok.load_state_dict(ckpt['model'])
cam_tok.eval()

# vid_tok = CausalVideoTokenizer(checkpoint_dec=f'./ckpt/decoder.jit')

toks = {
    'tok_cam': cam_tok,
    # 'tok_vid': vid_tok,
}

checkpoint = torch.load('./ckpt/checkpoint-main.pth', map_location='cpu', weights_only=False)
args = checkpoint['args']
model = get_model(args, setup_data(args)[0])
model.load_state_dict(checkpoint['model'])
model = model.eval().to(device)
sampler = GenerationSampler(model)

cond_domains = ['tok_rgb']
target_domains = ['tok_cam']
tokens_per_target = [30]
autoregression_schemes = ['roar'] * len(target_domains)

# NOTE: you can tune this parameter for best eval performance
decoding_steps = [3] * len(target_domains)
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

# here we give another example to directly do inference on tokenized rgb modalities
for file in glob.glob('example_data/rgb2cam_*.npz'): 
    name = os.path.basename(file)
    print(name)

    start = time.time()

    batched_sample = {
        'tok_rgb': {
            'tensor': torch.as_tensor(np.load(file)['arr_0']).unsqueeze(0).to(device), # Batched tensor
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

    dec_dict = decode_cam(
        name,
        out_dict, toks,
        image_size=256, patch_size=8,
        name = 'example_data/rgb2cam'
    )

    print(f'predicting {file}')

print(f'use vis_3d/vis_cam.py to visualize predicted camera trajectory!')