#!/bin/bash
#SBATCH --job-name=cam_opencv
#SBATCH --output reports/R-%x.%j.out    # Make sure this paths exists, otherwise the job will fail silently
#SBATCH --error reports/R-%x.%j.err     # Make sure this paths exists, otherwise the job will fail silently
#SBATCH --account=a144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --array=0-3%1
#SBATCH --time=12:00:00


echo "START TIME: $(date)"

# auto-fail on any errors in this script
# set -eo pipefail

# logging script's variables/commands for future debug needs
set -x
ulimit -c 0
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345

LAUNCHER="NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 numactl --membind=0-3 /iopsstor/scratch/cscs/lgen/pytorch-25.01-py3-venv/egom2p/bin/python -m torch.distributed.run \
    --nproc_per_node 4 \
    --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    --node_rank ${SLURM_PROCID} \
    "

export CMD="CUDA_DEVICE_MAX_CONNECTIONS=1 $LAUNCHER run_training_vqvae.py --config cfgs/default/tokenization/vqvae/cam_traj/Transformer_cam_traj_256_f60_cano_norm_ds2_slr3_opencv.yaml"

echo $CMD

SRUN_ARGS=" \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    --unbuffered \
    --environment=/users/lgen/.edf/egom2p.toml \
    --container-workdir=/users/lgen/egom2p \
    "

srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"

