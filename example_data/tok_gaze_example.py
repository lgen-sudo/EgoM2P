'''
Example to use pretrained gaze tokenizer to tokenize data
'''
import os
import numpy as np

# tokenize gaze from gaze.npy:
# TODO: ensure gaze shape 60x2, x is the horizontal coordinate and y is the vertical coordinate
# TODO: ensure gaze is already normed to 0-1 range with the image resolution! 
# TODO: we also assume square input image size. do the center crop by yourself! 
# you can check convert() and mode == 'tokenize' in egom2p/data/gaze_dataset.py
# NOTE: please check L108 for the preset params of gaze_dataset.py since this example gaze data is from holoassist dataset
# normally you should set orig_res, resize_res and new_res in convert().

# put the data path after --tokenize_path, the data is loaded in egom2p/data/gaze_dataset.py (search example in that file)

# save computed tokens to example_data/token/gaze-tok.npz
os.system('OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=1 run_training_vqvae.py \
            --config cfgs/default/tokenization/vqvae/gaze/Transformer_gaze_256_frame_60_ds2.yaml \
            --resume ckpt/checkpoint-gaze.pth \
            --tokenize --tokenize_path example_data/gaze.npy \
            --tokenize_save_path example_data/ --no_log_wandb')

# do autoencode and check reconstructed gaze dynamics is close to the input one
# save reconstructed gaze dynamics to example_data/token/gaze-recon.npy
os.system('OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=1 run_training_vqvae.py \
            --config cfgs/default/tokenization/vqvae/gaze/Transformer_gaze_256_frame_60_ds2.yaml \
            --resume ckpt/checkpoint-gaze.pth \
            --tokenize --recon --tokenize_path example_data/gaze.npy \
            --tokenize_save_path example_data/ --no_log_wandb')

def convert(gaze_data, orig_res, resize_res, new_res=[480, 480]):
    # convert gaze 2d coordinates in the original resolution to 480x480
    orig_res = np.array(orig_res)
    new_res = np.array(new_res)
    gaze_normed = gaze_data / orig_res # to [0, 1]
    gaze_resize_coord = gaze_normed * np.array(resize_res) # resized coord

    # check if valid in center cropped image (new_res)
    _min = (resize_res - new_res) / 2
    gaze_new_coord = gaze_resize_coord - _min
    gaze = gaze_new_coord / np.array(new_res)

    return gaze

print()
print('#######################################')
print('MSE for gaze_recon and gaze_input:')

# resize_res is the same with orig_res: no resize
# new_res: center crop to new resolution
converted_input = convert(np.load('example_data/gaze.npy'), orig_res=[896, 504], resize_res=[896, 504], new_res=[480, 480])
pred = np.load('example_data/token/gaze-recon.npy')

# holoassist dataset has nan values for gaze, while we can "denoise" nans when do reconstruction
mask = ~np.isnan(converted_input)
print('MSE: ', ((converted_input[mask] - pred[mask]) ** 2).mean())
