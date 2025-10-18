'''
Example to use pretrained camera tokenizer to tokenize data (use cam.npy as example)
'''
import os
import numpy as np
from scipy.spatial.transform import Rotation

# tokenize camera extrinsics from cam.npy:
# TODO: ensure camera pose shape of 60x4x4, in opencv convention
# put the data path after --tokenize_path, the data is loaded in egom2p/data/cam_traj_dataset.py

# save computed tokens to example_data/token/cam-tok.npz
os.system('OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=1 run_training_vqvae.py \
            --config cfgs/default/tokenization/vqvae/cam_traj/Transformer_cam_traj_256_f60_cano_norm_ds2_slr3_opencv.yaml \
            --resume ckpt/checkpoint-cam.pth \
            --tokenize --tokenize_path example_data/cam.npy \
            --tokenize_save_path example_data/ --no_log_wandb')

# do autoencode and check reconstructed camera pose is close to the input one
# save reconstructed camera pose to example_data/token/cam-recon.npy
os.system('OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=1 run_training_vqvae.py \
            --config cfgs/default/tokenization/vqvae/cam_traj/Transformer_cam_traj_256_f60_cano_norm_ds2_slr3_opencv.yaml \
            --resume ckpt/checkpoint-cam.pth \
            --tokenize --recon --tokenize_path example_data/cam.npy \
            --tokenize_save_path example_data/ --no_log_wandb')

print()
print('###########################################')
print('starting to check if the reconstructed camera traj is close to the input traj')
print()

def canonicalize(sample):
    inv = np.linalg.inv(sample[0])
    canoed = np.einsum('ij, kjl -> kil', inv, sample)
    # In 6D representation for rotation matrix, we take the first two columns
    rot6d = canoed[:, :3, :2] 
    transl = canoed[:, :3, 3:]
    # Note: The original 6D representation is column-major.
    # So we reshape to (N, 6) and concatenate with translation (N, 3)
    cam_9d = np.concatenate(
        (rot6d.transpose(0, 2, 1).reshape(-1, 6), transl.squeeze(-1)),
        axis=-1
    )
    return cam_9d

def convert_9d_to_4x4(traj_9d):
    """
    Converts a trajectory in 9D format (N, 9) into a series of
    4x4 homogeneous matrices (N, 4, 4).

    Args:
        traj_9d (np.ndarray): A trajectory of shape (N, 9).

    Returns:
        np.ndarray: The trajectory as 4x4 matrices of shape (N, 4, 4).
    """
    if traj_9d.ndim == 1 and traj_9d.shape[0] % 9 == 0:
        traj_9d = traj_9d.reshape(-1, 9)
    elif traj_9d.ndim > 2:
        raise ValueError("Input traj_9d must be of shape (N, 9) or flattened")

    # Isolate the 6D rotation and 3D translation parts
    d6 = traj_9d[..., :6]
    transl = traj_9d[..., 6:]

    # Extract the two raw vectors for the rotation matrix
    a1, a2 = d6[..., :3], d6[..., 3:]

    # Gram-Schmidt orthogonalization
    b1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = np.divide(a1, b1_norm, out=np.zeros_like(a1), where=b1_norm != 0)

    dot_product = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot_product * b1
    b2_norm = np.linalg.norm(b2, axis=-1, keepdims=True)
    b2 = np.divide(b2, b2_norm, out=np.zeros_like(b2), where=b2_norm != 0)

    b3 = np.cross(b1, b2, axis=-1)
    rotmat = np.stack((b1, b2, b3), axis=-1)

    # Create the final 4x4 homogeneous transformation matrices
    num_poses = traj_9d.shape[0]
    mats = np.zeros((num_poses, 4, 4))
    mats[:, :3, :3] = rotmat
    mats[:, :3, 3] = transl
    mats[:, 3, 3] = 1.0

    return mats

def calculate_ate(pred_traj_4x4, gt_traj_4x4):
    """
    Calculates the Absolute Trajectory Error (ATE).
    This function aligns the predicted trajectory to the ground truth
    trajectory using the Umeyama algorithm and then computes the
    root-mean-squared error between the translations.

    Args:
        pred_traj_4x4 (np.ndarray): Predicted trajectory of shape (N, 4, 4).
        gt_traj_4x4 (np.ndarray): Ground truth trajectory of shape (N, 4, 4).

    Returns:
        float: The ATE (RMSE) value.
    """
    # 1. Extract translation vectors
    pred_t = pred_traj_4x4[:, :3, 3]
    gt_t = gt_traj_4x4[:, :3, 3]

    # 2. Find the optimal alignment (rotation R) using Umeyama/Kabsch algorithm
    #    This aligns the centered predicted points to the centered ground truth points.
    pred_t_centered = pred_t - pred_t.mean(axis=0)
    gt_t_centered = gt_t - gt_t.mean(axis=0)
    
    # Use SVD to find the rotation. This is the core of the algorithm.
    W = np.dot(gt_t_centered.T, pred_t_centered)
    U, _, Vt = np.linalg.svd(W)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0: # Ensure a right-handed coordinate system
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # 3. Apply the alignment to the predicted trajectory
    pred_t_aligned = np.dot(R, pred_t.T).T

    # 4. Calculate the error and the final ATE (RMSE)
    alignment_error = gt_t - pred_t_aligned
    ate = np.sqrt(np.mean(np.sum(alignment_error**2, axis=1)))

    return ate

def calculate_rpe(pred_traj_4x4, gt_traj_4x4, delta=1):
    """
    Calculates the Relative Pose Error (RPE), which includes both
    Relative Translation Error (RTE) and Relative Rotation Error (RRE).
    It computes the error in the transformation between consecutive poses.

    Args:
        pred_traj_4x4 (np.ndarray): Predicted trajectory of shape (N, 4, 4).
        gt_traj_4x4 (np.ndarray): Ground truth trajectory of shape (N, 4, 4).
        delta (int): The step size between poses to compute the relative motion.

    Returns:
        tuple[float, float]: A tuple containing (RTE, RRE_deg).
                             RTE is the RMSE of the translational part.
                             RRE_deg is the RMSE of the rotational part in degrees.
    """
    if pred_traj_4x4.shape[0] != gt_traj_4x4.shape[0]:
        raise ValueError("Trajectories must have the same length.")

    # 1. Calculate relative poses for both trajectories
    gt_inv = np.linalg.inv(gt_traj_4x4[:-delta])
    gt_rel = gt_inv @ gt_traj_4x4[delta:]

    pred_inv = np.linalg.inv(pred_traj_4x4[:-delta])
    pred_rel = pred_inv @ pred_traj_4x4[delta:]

    # 2. Calculate the error matrix for each pair of relative poses
    #    error = (gt_relative)^-1 * pred_relative
    error_mat = np.linalg.inv(gt_rel) @ pred_rel

    # 3. Extract translational and rotational errors
    trans_errors = error_mat[:, :3, 3]
    rot_errors_mat = error_mat[:, :3, :3]

    # 4. Calculate RTE (RMSE of translation norms)
    rte = np.sqrt(np.mean(np.sum(trans_errors**2, axis=1)))

    # 5. Calculate RRE (RMSE of rotation angles)
    # Convert rotation matrices to rotation vectors, where the magnitude
    # is the angle of rotation in radians.
    rot_errors_vec = Rotation.from_matrix(rot_errors_mat).as_rotvec()
    rot_angles_rad = np.linalg.norm(rot_errors_vec, axis=1)
    rre_rad = np.sqrt(np.mean(rot_angles_rad**2))
    rre_deg = np.rad2deg(rre_rad)

    return rte, rre_deg

def evaluate_9d_trajectory(pred_traj_9d, gt_traj_9d, delta=1):
    """
    A wrapper function to calculate ATE, RTE, and RRE from 9D trajectories.

    Args:
        pred_traj_9d (np.ndarray): Predicted trajectory in 9D format (N, 9).
        gt_traj_9d (np.ndarray): Ground truth trajectory in 9D format (N, 9).
        delta (int): Step size for RPE calculation.

    Returns:
        dict: A dictionary containing the error metrics.
    """
    # First, convert both trajectories to 4x4 matrix format
    pred_4x4 = convert_9d_to_4x4(pred_traj_9d)
    gt_4x4 = convert_9d_to_4x4(gt_traj_9d)
    
    # Calculate metrics
    ate_val = calculate_ate(pred_4x4, gt_4x4)
    rte_val, rre_val_deg = calculate_rpe(pred_4x4, gt_4x4, delta=delta)

    return {
        "ATE": ate_val,
        "RTE": rte_val,
        "RRE_deg": rre_val_deg
    }

cano_input = canonicalize(np.load('example_data/cam.npy'))
recon = np.load('example_data/token/cam-recon.npy') # generated by the above command

# you will see the output (meter for ATE and RTE, degree for RRE): 
# {'ATE': 0.0058740415749732955, 'RTE': 0.002837537725354713, 'RRE_deg': 0.2980470115530609}
# this camera traj is from EgoGen data, the error is roughly on par with Tab C.1 in the supp mat
print(evaluate_9d_trajectory(recon, cano_input, delta=1))