# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A CLI to run CausalVideoTokenizer on plain videos based on torch.jit.

Usage:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --output_dir ./reconstructions \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit

    Optionally, you can run the model in pure PyTorch mode:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --mode=torch \
        --tokenizer_type=CV \
        --temporal_compression=4 \
        --spatial_compression=8 \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any
import sys
import tarfile
from io import BytesIO
import numpy as np
from loguru import logger as logging
import pdb
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.utils import (
    get_filepaths,
    get_output_filepath,
    read_video,
    resize_video,
    write_video,
)
import tempfile
import mediapy as mp
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import tqdm
import glob
import cv2
import subprocess
from multiprocessing import Pool
import multiprocessing

def _parse_args() -> tuple[Namespace, dict[str, Any]]:
    parser = ArgumentParser(description="A CLI for CausalVideoTokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="path/to/videos/*.mp4",
        help="Glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="JIT full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default='pretrained_ckpts/Cosmos-Tokenizer-DV8x16x16/encoder.jit',
        help="JIT Encoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="JIT Decoder model filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CV", "DV"],
        help="Specifies the tokenizer type.",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=16,
        help="The spatial compression factor.",
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[4, 8],
        default=8,
        help="The temporal compression factor.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="jit",
        help="Specify the backend: native 'torch' or 'jit' (default: 'jit')",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=256,
        help="The size to resample inputs. None, by default.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=17,
        help="The temporal window to operate at a time.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Sets the precision, default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for invoking the model.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=8.0,
        help="Output frames-per-second (FPS).",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="If on, the input video will be be outputted too.",
    )

    args = parser.parse_args()
    return args


def _run_eval(filepath, args, gpu_id) -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""
    # print(filepath, gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if (
        args.checkpoint_enc is None
        and args.checkpoint_dec is None
        and args.checkpoint is None
    ):
        logging.warning(
            "Aborting. Both encoder or decoder JIT required. Or provide the full autoencoder JIT model."
        )
        return

    if args.mode == "torch":
        tokenizer_config = TokenizerConfigs[args.tokenizer_type].value
        tokenizer_config.update(dict(spatial_compression=args.spatial_compression))
        tokenizer_config.update(dict(temporal_compression=args.temporal_compression))
    else:
        tokenizer_config = None

    # logging.info(
    #     f"Loading a torch.jit model `{os.path.dirname(args.checkpoint or args.checkpoint_enc or args.checkpoint_dec)}` ..."
    # )
    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )
    
    # logging.info(f"Found {len(filepaths)} videos.")
    out_path = get_output_filepath(filepath, output_dir=args.output_dir)
    if os.path.exists(out_path):
        return
    print(filepath, flush=True)
    with tarfile.open(filepath, "r") as tar:
        tar_out = tarfile.open(out_path, 'w')
        for member in tqdm.tqdm(tar.getmembers()):
            if member.isfile() and member.name.endswith('.mp4'):
                file_content = tar.extractfile(member).read()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                    temp_input_path = temp_input.name
                    temp_input.write(file_content)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                    temp_output_path = temp_output.name

                # downsample video to 8 fps and center crop and resize
                crop_size = 480
                resize_size = 256

                # Get video dimensions using OpenCV
                cap = cv2.VideoCapture(temp_input_path)
                input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                
                # Calculate crop dimensions
                x_offset = (input_width - crop_size) // 2
                y_offset = (input_height - crop_size) // 2

                ffmpeg_command = [
                    'ffmpeg', 
                    '-i', temp_input_path,  # Input file
                    '-vf', f"fps={args.output_fps},crop={crop_size}:{crop_size}:{x_offset}:{y_offset},scale={resize_size}:{resize_size}:flags=lanczos",
                    '-c:v', 'libx264',  # Video codec
                    '-crf', '18',
                    '-preset', 'slow',
                    '-y',  # Overwrite output file
                    temp_output_path
                ]
                subprocess.run(
                    ffmpeg_command, 
                    capture_output=True, 
                    text=True
                )
                with open(temp_output_path, 'rb') as f:
                    downsampled_content = f.read()

                video = mp.decompress_video(downsampled_content)
                # write_video(f'./{member.name}.mp4', video, fps=8)
                # exit(-1)
                # logging.info("Invoking the autoencoder model in ... ")
                batch_video = video[np.newaxis, ...]
                # output_video = autoencoder(batch_video, temporal_window=args.temporal_window)[0]
                # write_video('./output.mp4', output_video, fps=8)
                
                output_token = autoencoder(batch_video, temporal_window=args.temporal_window)[0]
                # logging.info("Constructing output filepath ...")
                # write_video(output_filepath, output_video, fps=args.output_fps)
                bytes_io1 = BytesIO()
                np.savez_compressed(bytes_io1, output_token)
                bytes_io1.seek(0)
                filename = os.path.basename(member.name)
                if '.mp4' in filename:
                    filename = filename.replace('.mp4', '.npz')
                file_data = bytes_io1.getvalue()
                tarinfo1 = tarfile.TarInfo(name=filename)
                tarinfo1.size = len(file_data)
                # Write to the tarfile
                tar_out.addfile(tarinfo1, BytesIO(file_data))

        tar_out.close()


def check_and_delete_tar(tar_path):
    try:
        # Attempt to open the tar file and list its contents
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()
    except (tarfile.TarError, EOFError, IOError) as e:
        # If an error occurs, the tar file is considered invalid
        print(f"Error: The tar file '{tar_path}' is invalid. Deleting the file.")
        os.remove(tar_path)

def worker(gpu_id, file_paths, args):
    for file_path in file_paths:
        _run_eval(file_path, args, gpu_id)


if __name__ == "__main__":
    # ctx = mp.get_context('spawn')
    x = []
    # x.extend(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/video_tar/train*.tar'))
    # x.extend(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/video_tar/val*.tar'))
    # x.extend(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/video_tar/*.tar'))
    # x.extend(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/h2o/video_tar/*.tar'))
    x.extend(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/taco/video_tar/*.tar'))
    # start = 4
    filepaths = sorted(x)# [start:]# [start*3:(start+1)*3]
    chunks = [filepaths[i::4] for i in range(4)]
    args = _parse_args()

    processes = []
    for gpu_id in range(4):
        p = multiprocessing.Process(target=worker, args=(gpu_id, chunks[gpu_id], args))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # files = glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/depth/holoassist/video_tar/*.tar')
    # with Pool(120) as pool:
    #     pool.map_async(check_and_delete_tar, files)
