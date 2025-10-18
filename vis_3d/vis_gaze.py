import numpy as np
import argparse
import pdb
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import tqdm
import sys
import shutil
from fractions import Fraction

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='/capstor/scratch/cscs/lgen/datasets_aligned/gaze/holoassist/R0027-12-GoPro_10.npy')
parser.add_argument('--recon', type=str, default='/capstor/users/cscs/lgen/egom2p/eval_model_res/R0027-12-GoPro_10.npz_tok_gaze.npy')
parser.add_argument('--quant', type=str, default='/capstor/users/cscs/lgen/egom2p/eval_model_res/R0027-12-GoPro_10.npz_gaze_quantized_gt.npy')
parser.add_argument('--vid', type=str, default='/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/video_tar/tmp/R0027-12-GoPro_10.mp4')
args = parser.parse_args()

gaze_gt = np.load(args.gt) # in original resolution. may contain nan. need to convert it to cropped origin.
gaze_recon = np.load(args.recon) # in 0-1. 
gaze_quant = np.load(args.quant)

cap = cv2.VideoCapture(args.vid)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_out = cv2.VideoWriter(os.path.join('/capstor/users/cscs/lgen/egom2p/eval_model_res', 'gaze_%s.mp4' % os.path.basename(args.vid)), fourcc, frame_rate, (480, 480), isColor=True)

num_frames = len(gaze_gt)

for frame_idx in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Center crop the video to 480x480
    h, w, _ = frame.shape
    start_x = (w - 480) // 2
    start_y = (h - 480) // 2
    cropped_frame = frame[start_y:start_y + 480, start_x:start_x + 480]
    
    # Process ground truth points
    points_gt = gaze_gt[frame_idx] - np.array([208, 12])
    draw_gt = not np.isnan(points_gt).any()  # Determine if gt should be drawn
    
    # Process reconstructed points
    points_pred = gaze_recon[frame_idx] * np.array([480, 480])
    points_quant = gaze_quant[frame_idx] * np.array([480, 480])
    
    # Draw circles on the cropped video
    if draw_gt:
        thickness = 2
        radius = 4
        color_gt = (0, 0, 255)
        cv2.circle(cropped_frame, (int(points_gt[0]), int(points_gt[1])), radius, color_gt, thickness)
    
    thickness = 2
    radius = 2
    color_pred = (255, 0, 0)
    cv2.circle(cropped_frame, (int(points_pred[0]), int(points_pred[1])), radius, color_pred, thickness)

    thickness = 2
    radius = 2
    color_pred = (0, 255, 0)
    cv2.circle(cropped_frame, (int(points_quant[0]), int(points_quant[1])), radius, color_pred, thickness)
    
    # Write the processed frame to the output video
    video_out.write(cropped_frame)

# Release resources
cap.release()
video_out.release()
cv2.destroyAllWindows()