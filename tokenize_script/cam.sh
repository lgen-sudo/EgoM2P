
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 run_training_vqvae.py \
                        --config cfgs/default/tokenization/vqvae/cam_traj/Transformer_cam_traj_256_f60_cano_norm_ds2_slr3.yaml \
                        --resume /iopsstor/scratch/cscs/lgen/output/tokenization/vqvae/cam_traj/checkpoint-best.pth \
                        --tokenize --tokenize_path /capstor/scratch/cscs/lgen/datasets_aligned/hot3d \
                        --tokenize_save_path /iopsstor/scratch/cscs/lgen/main_model_token/cam/hot3d --no_log_wandb
