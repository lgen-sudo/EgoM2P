import argparse
import os
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from cam_viz_tool import SLAMFrontend
import pdb

# NOTE: please refer to nice-slam codebase for installing all necessary dependencies!

def _9d_to_mat(_9d):
    d6 = _9d[..., :6]
    transl = _9d[..., 6:]
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rotmat = torch.stack((b1, b2, b3), dim=-2).permute(0, 2, 1)
    mat = torch.zeros(d6.shape[0], 4, 4)
    mat[:, :3, :3] = rotmat
    mat[:, :3, 3] = transl
    mat[:, 3, 3] = 1.
    return mat.numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('--output', type=str, default='cam_tok_res',
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    parser.add_argument('--recon', type=str, default='/capstor/users/cscs/lgen/egom2p/eval_model_res/R0027-12-GoPro_10.npz_tok_cam.npy')
    parser.add_argument('--gt', type=str, default='/capstor/scratch/cscs/lgen/datasets_aligned/cam/holoassist/tmp/R0027-12-GoPro_10.npy')
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output = args.output
    
    estimate_c2w_list = _9d_to_mat(torch.as_tensor(np.load(args.recon)))
    gt_c2w_list = _9d_to_mat(torch.as_tensor(np.load(args.gt)))

    frontend = SLAMFrontend(output, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            save_rendering=args.save_rendering, near=0,
                            estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()

    for i in tqdm(range(0, gt_c2w_list.shape[0])):
        time.sleep(0.03)
        frontend.update_pose(1, estimate_c2w_list[i], gt=False)
        frontend.update_pose(1, gt_c2w_list[i], gt=True)
        frontend.update_cam_trajectory(i, gt=False)
        frontend.update_cam_trajectory(i, gt=True)

    if args.save_rendering:
        time.sleep(1)
        os.system(
            f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")