
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 run_training_vqvae.py \
                        --config cfgs/default/tokenization/vqvae/gaze/Transformer_gaze_256_frame_60_ds2.yaml \
                        --resume /iopsstor/scratch/cscs/lgen/output/tokenization/vqvae/gaze/checkpoint-best.pth \
                        --tokenize --tokenize_path /capstor/scratch/cscs/lgen/datasets_aligned/hot3d \
                        --tokenize_save_path /iopsstor/scratch/cscs/lgen/main_model_token/gaze/hot3d --no_log_wandb
