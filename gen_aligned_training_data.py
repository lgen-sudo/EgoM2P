import numpy as np
import glob
import os
import cv2
from fractions import Fraction
from sklearn.neighbors import NearestNeighbors
from decord import VideoReader
from decord import cpu
import pdb
import tarfile
import io
import subprocess
import tqdm
from multiprocessing import Pool
import gc
import fnmatch
import re

def get_shard_idx(path):
    all_shards = glob.glob(os.path.join(path, 'shard-*.tar'))
    latest_shards = -1
    for shard in all_shards:
        t = shard.split('-')[-1].split('.')[0]
        if t.isdigit():
            latest_shards = max(int(t), latest_shards)
    return latest_shards + 1

def holo_cam():

    def read_pose(data_path):
        img_pose_array = []
        with open(data_path) as f:
            lines = f.read().split('\n')
            for line in lines:
                if line == '':  # end of the lines.
                    break
                line_data = list(map(float, line.split('\t')))
                img_pose_array.append(line_data)
            img_pose_array = np.array(img_pose_array)[:, 2:].reshape(-1, 4, 4)
        return img_pose_array
    
    out_path = os.path.join(aligned_path, 'cam', 'holoassist')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_path = '/capstor/store/cscs/swissai/a03/datasets/holoassist'
    all_list = []
    with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'train-v1_2.txt')) as f:
        train_list = f.read().splitlines()
    with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'val-v1_2.txt')) as f:
        val_list = f.read().splitlines()
    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'test-v1_2.txt')) as f:
    #     test_list = f.read().splitlines()
    all_list.extend(train_list)
    all_list.extend(val_list)
    # all_list.extend(test_list)
    for video_name in all_list:
        print(video_name)
        base_path = os.path.join(in_path, video_name, "Export_py")
        # data_path = os.path.join(self.data_path, 'video_compress', x, 'Export_py', 'Video', 'Pose_sync.txt')
        data_path = os.path.join(base_path, 'Video', 'Pose_sync.txt')
        cam_traj = read_pose(data_path)
        for start_index in range(0, len(cam_traj), NUM_FRAMES_PER_SAMPLE):
            if start_index + NUM_FRAMES_PER_SAMPLE > len(cam_traj):
                break
            cam_pose = cam_traj[start_index : start_index + NUM_FRAMES_PER_SAMPLE]
            np.save(os.path.join(out_path, f"{video_name}_{start_index // NUM_FRAMES_PER_SAMPLE}.npy"), cam_pose)

def holo_gaze():

    def read_gaze_txt(gaze_path):
        with open(gaze_path) as f:
            gaze_data = []
            lines = f.read().split('\n')
            for line in lines:
                if line == '':  # end of the lines.
                    break
                line_data = list(map(float, line.strip().split('\t')))
                gaze_data.append(line_data)
            gaze_data = np.array(gaze_data)[:, 2:]
        return gaze_data
    
    out_path = os.path.join(aligned_path, 'gaze', 'holoassist')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_path = '/capstor/store/cscs/swissai/a03/datasets/holoassist'
    all_list = []
    with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'train-v1_2.txt')) as f:
        train_list = f.read().splitlines()
    with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'val-v1_2.txt')) as f:
        val_list = f.read().splitlines()
    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'test-v1_2.txt')) as f:
    #     test_list = f.read().splitlines()
    all_list.extend(train_list)
    all_list.extend(val_list)
    # all_list.extend(test_list)
    for video_name in tqdm.tqdm(all_list):
        print(video_name)
        base_path = os.path.join(in_path, video_name, "Export_py")
        data_path = os.path.join(base_path, 'Eyes', 'Eyes_proj.txt')
        gaze_data = read_gaze_txt(data_path)
        # self.dataset_samples.append(self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504]))
        for start_index in range(0, len(gaze_data), NUM_FRAMES_PER_SAMPLE):
            if start_index + NUM_FRAMES_PER_SAMPLE > len(gaze_data):
                break
            gaze = gaze_data[start_index : start_index + NUM_FRAMES_PER_SAMPLE]
            np.save(os.path.join(out_path, f"{video_name}_{start_index // NUM_FRAMES_PER_SAMPLE}.npy"), gaze)

def cut_video(video_name):

    out_path = os.path.join(aligned_path, 'rgb', 'holoassist')
    in_path = '/capstor/store/cscs/swissai/a03/datasets/holoassist'

    # print(video_name)
    base_path = os.path.join(in_path, video_name, "Export_py")
    img_path = os.path.join(base_path, 'Video')

    # Read timing file
    img_sync_timing_path = os.path.join(img_path, 'Pose_sync.txt')
    img_sync_timing_array = []
    with open(img_sync_timing_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = int(line.split('\t')[1])
            img_sync_timing_array.append(line_data)

    start_time_path = os.path.join(img_path, 'VideoMp4Timing.txt')
    with open(start_time_path) as f:
        lines = f.read().split('\n')
        start_time = int(lines[0])
    
    cap = cv2.VideoCapture(os.path.join(base_path,"Video_pitchshift.mp4"))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_name, frame_num // NUM_FRAMES_PER_SAMPLE)

    frame_rate_fraction = Fraction(frame_rate).limit_denominator()
    img_timing_array = []
    for ii in range(frame_num):
        # FIX 2: reduce floating point calculation error.
        # img_timing_array.append(int(start_time + ii * (1/frame_rate)* 10**7))
        frame_ticks = (ii * frame_rate_fraction.denominator * 10**7) // frame_rate_fraction.numerator
        img_timing_array.append(start_time + frame_ticks)
    
    timestamp_array_mepg = np.array(img_timing_array)
    mpeg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
            timestamp_array_mepg.reshape(-1, 1))
    
    ori_frame_idx = [] # in new video, each frame corresponds to the original frame index
    for ii , img_sync_timestamp in enumerate(img_sync_timing_array[:]):
        _, mpeg_indices = mpeg_nbrs.kneighbors(
            np.array(img_sync_timestamp).reshape(-1, 1))
        ori_frame_idx.append(mpeg_indices[0][0])
    ori_frame_idx = np.array(ori_frame_idx)
    
    vr = VideoReader(os.path.join(base_path,"Video_pitchshift.mp4"), ctx=cpu(0))
    vid_len = len(vr)
    video_data = vr.get_batch(range(vid_len)).asnumpy()

    for start_index in range(0, len(ori_frame_idx), NUM_FRAMES_PER_SAMPLE):
        if start_index + NUM_FRAMES_PER_SAMPLE > len(ori_frame_idx):
            break
        ori_vid_index = ori_frame_idx[start_index : start_index + NUM_FRAMES_PER_SAMPLE]
        video_sample = video_data[ori_vid_index]
        _, height, width, _ = video_sample.shape

        # Create a temporary mp4 file path
        mp4_filename = os.path.join(out_path, f"{video_name}_{start_index // NUM_FRAMES_PER_SAMPLE}.mp4")

        # Launch FFmpeg with desired encoding parameters
        process = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(30), 
            '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'slow', '-loglevel', 'error',
            '-crf', '18', '-f', 'mp4', mp4_filename
        ], stdin=subprocess.PIPE)

        for frame in video_sample:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
    del video_data
    del mpeg_nbrs
    gc.collect()
    np.savetxt(os.path.join(out_path, f'{video_name}-done.txt'), np.array([1]))
    print(video_name, 'done')

def tar_holo(modality):
    out_path = os.path.join(aligned_path, "%s" % modality)
    in_path = '/capstor/store/cscs/swissai/a03/datasets/holoassist'
    # in_path = '/store/swissai/a03/datasets/holoassist'
    TAR_IDX = get_shard_idx(out_path)
    file_count = 0
    tar = None  # Placeholder for the tar file object  

    with open(os.path.join(in_path, 'data_split', 'train-v1_2.txt')) as f:
        train_list = f.read().splitlines()

    # for video_name in tqdm.tqdm(sorted(train_list)):
    #     print(video_name)
    #     for token in sorted(glob.glob(os.path.join(out_path, 'holoassist', 'token', f'{video_name}*.npz'))):
    #         if file_count % NUM_FILES_PER_TAR == 0:
    #             if tar is not None:
    #                 tar.close()

    #             # Create a new tar file
    #             tar_filename = f'shard-{TAR_IDX:06d}.tar'
    #             tar = tarfile.open(os.path.join(out_path, tar_filename), 'w')
    #             print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
    #             TAR_IDX += 1
    #         filename = os.path.basename(token)
    #         if '.mp4' in filename:
    #             filename = filename.replace('.mp4', '')
    #         tar.add(token, arcname=filename)
    #         file_count += 1

    for token in tqdm.tqdm(sorted(glob.glob(os.path.join(out_path, 'holoassist', 'token', '*.npz')))):
        video = re.sub(r'_\d+\.npz', '', os.path.basename(token))
        if video in train_list:
            if file_count % NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'shard-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(out_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            filename = os.path.basename(token)
            if '.mp4' in filename:
                filename = filename.replace('.mp4', '')
            tar.add(token, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)


def tar_files(in_path):
    TAR_IDX = 0
    file_count = 0
    tar = None  # Placeholder for the tar file object

    with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'train-v1_2.txt')) as f:
        train_list = f.read().splitlines()

    for token in tqdm.tqdm(sorted(glob.glob(os.path.join(in_path, '*.npz')))):
        video = re.sub(r'_\d+_pred\.npz', '', os.path.basename(token))
        if video in train_list:
            if file_count % NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'train-dep-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(in_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            filename = os.path.basename(token)
            tar.add(token, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)


def tar_files_hot3d(in_path="/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d"):
    TAR_IDX = 0
    file_count = 0
    tar = None  # Placeholder for the tar file object

    test_list = ["P0004", "P0005", "P0006", "P0008", "P0016", "P0020"]
    val_list = ["P0001", "P0014"]

    for mp4 in tqdm.tqdm(sorted(glob.glob(os.path.join(in_path, 'video_ds', '*.mp4')))):
        filename = os.path.basename(mp4)
        subject = filename.split('_')[0]

        if subject not in val_list and subject not in test_list:
            
            if file_count % NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'train-mp4-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(in_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            tar.add(mp4, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)


def tar_files_taco(in_path="/capstor/scratch/cscs/lgen/datasets_aligned/depth/taco"):
    TAR_IDX = 0
    file_count = 0
    tar = None  # Placeholder for the tar file object

    val_idx = np.linspace(0, 2489, 500).astype(int)

    for mp4 in tqdm.tqdm(sorted(glob.glob(os.path.join(in_path, '*.npz')))):
        filename = os.path.basename(mp4)
        idx = int(os.path.splitext(filename)[0])

        if idx in val_idx:
            if file_count % NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'val-pred-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(in_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            tar.add(mp4, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)

def tar_files_h2o(in_path="/capstor/scratch/cscs/lgen/datasets_aligned/rgb/h2o"):
    TAR_IDX = 0
    file_count = 0
    tar = None  # Placeholder for the tar file object

    val_idx = ['subject3_ego-k2', 'subject3_ego-o1', 'subject3_ego-o2']

    for mp4 in tqdm.tqdm(sorted(glob.glob(os.path.join(in_path, 'video_ds', '*.mp4')))):
        filename = os.path.basename(mp4)
        seq_name = os.path.splitext(filename)[0][:-5]
        if seq_name in val_idx:
            if file_count % NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'val-mp4-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(in_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            tar.add(mp4, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)

def tar_files_arctic(in_path="/capstor/scratch/cscs/lgen/datasets_aligned/depth/arctic/"):
    TAR_IDX = 0
    _NUM_FILES_PER_TAR = 750
    file_count = 0
    tar = None  # Placeholder for the tar file object

    val_idx = 's05'

    for mp4 in tqdm.tqdm(sorted(glob.glob(os.path.join(in_path, 'pred', '*.npz')))):
        filename = os.path.basename(mp4)
        seq_name = filename.split('_')[0]
        if seq_name != val_idx:
            if file_count % _NUM_FILES_PER_TAR == 0:
                if tar is not None:
                    tar.close()

                # Create a new tar file
                tar_filename = f'train-pred-{TAR_IDX:06d}.tar'
                tar = tarfile.open(os.path.join(in_path, tar_filename), 'w')
                print(f'TAR idx: {TAR_IDX}, MP4 idx: {file_count}')
                TAR_IDX += 1
            tar.add(mp4, arcname=filename)
            file_count += 1
        
    if tar is not None:
        tar.close()

    print(file_count)
    print(TAR_IDX)


def downsample_video(input_file):
    # for holoassist, center-crop 480x480 and then resize it to 224x224
    file_name = os.path.basename(input_file)
    print(file_name)
    output_file = os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb_32x224x224/video_clips', file_name)
    # Center crop to 480x480 and resize to 224x224
    crop_size = 480
    resize_size = 224
    # output_file = f'./{resize_size}-{file_name}'

    # Input video resolution (width=896, height=504)
    input_width = 896
    input_height = 504

    # Calculate crop dimensions
    x_offset = (input_width - crop_size) // 2
    y_offset = (input_height - crop_size) // 2

    # Build the ffmpeg command
    # command = (
    #     f"ffmpeg -i {input_file} "
    #     f"-vf \"select='not(mod(n\\,{120 // frame_count}))',crop={crop_size}:{crop_size}:{x_offset}:{y_offset},scale={resize_size}:{resize_size}\" "
    #     f"-vcodec libx264 -crf 18 -pix_fmt yuv420p {output_file}"
    # )
    command = (
        f"ffmpeg -i \"{input_file}\" "
        f"-vf \"fps=8,crop={crop_size}:{crop_size}:{x_offset}:{y_offset},scale={resize_size}:{resize_size}\" "
        f"-vcodec libx264 -crf 18 -pix_fmt yuv420p \"{output_file}\""
        f"> /dev/null 2>&1"
    )

    # Run the command
    ret = os.system(command)
    # if ret == 0:
    #     os.remove(input_file)


def extract_filenames_from_tar(tar_path):
    """Extracts file names from a tar file."""
    with tarfile.open(tar_path, 'r') as tar:
        return [member.name for member in tar.getmembers()]

def find_matching_files(file_names, search_dir='/capstor/scratch/cscs/lgen/datasets_aligned/depth/holoassist'):
    """Finds files in the given directory matching any of the specified file names."""
    matches = []
    for root, _, files in os.walk(search_dir):
        for file_name in file_names:
            for filename in fnmatch.filter(files, os.path.basename(file_name).replace('.mp4', '.npz')):
                matches.append(os.path.join(root, filename))
    return matches

def create_new_tar(original_tar, matching_files, output_dir):
    """Creates a new tar file with matching files."""
    new_tar_name = os.path.join(output_dir, os.path.basename(original_tar).replace('.tar', '_dep_pred.tar'))
    with tarfile.open(new_tar_name, 'w') as new_tar:
        for file_path in matching_files:
            new_tar.add(file_path, arcname=os.path.basename(file_path))
    print(f"Created {new_tar_name} with {len(matching_files)} files.")

def process_tar_file(tar_file, tar_dir, output_dir):
    """Processes a single tar file."""
    print(f"Processing {tar_file}...")
    tar_path = os.path.join(tar_dir, tar_file)
    file_names = extract_filenames_from_tar(tar_path)
    matching_files = find_matching_files(file_names)
    create_new_tar(tar_path, matching_files, output_dir)

def process_tar_files(tar_dir, output_dir):
    """
    Processes tar files in the specified directory to create new tar files
    with files that have the same names as those in the original tar files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of tar files in the directory
    tar_files = [f for f in os.listdir(tar_dir) if 'val' in f]

    # Use multiprocessing to process tar files concurrently
    with Pool(128) as pool:
        # Pass tar_dir and output_dir as additional arguments
        pool.starmap(process_tar_file, [(tar_file, tar_dir, output_dir) for tar_file in tar_files])

if __name__ == '__main__':
    aligned_path = '/capstor/scratch/cscs/lgen/datasets_aligned'
    # aligned_path = '/iopsstor/scratch/cscs/lgen/tmp'
    NUM_FILES_PER_TAR = 1000
    NUM_FRAMES_PER_SAMPLE = 120

    # # cut video
    # all_list = []
    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'train-v1_2.txt')) as f:
    #     train_list = f.read().splitlines()

    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'val-v1_2.txt')) as f:
    #     val_list = f.read().splitlines()
    # for video_name in val_list:
    #     os.system(f'mv /capstor/scratch/cscs/lgen/datasets_aligned/cam/holoassist/token/{video_name}*.npz /capstor/scratch/cscs/lgen/datasets_aligned/val/cam/holoassist/')
    #     os.system(f'mv /capstor/scratch/cscs/lgen/datasets_aligned/gaze/holoassist/token/{video_name}*.npz /capstor/scratch/cscs/lgen/datasets_aligned/val/gaze/holoassist/')

    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'test-v1_2.txt')) as f:
    #     test_list = f.read().splitlines()
    # all_list.extend(train_list)
    # all_list.extend(val_list)
    # all_list.extend(test_list)

    # todo_list = []
    # for video_name in all_list:
    #     if not os.path.exists(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist', f'{video_name}-done.txt')):
    #         todo_list.append(video_name)
    #     else:
    #         data = np.loadtxt(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist', f'{video_name}-done.txt'))
    #         if data != 1:
    #             todo_list.append(video_name)
    # todo_list = sorted(todo_list)
    # p = Pool(5)
    # p.map_async(cut_video, todo_list)
    # p.close()
    # p.join()
    
    # print('finished!!!!!')
    # print()

    # order: [holo, hot3d, ego4d, egoexo4d, adt, nymeria, ]
    # gen_holo_rgb()
    # read_hot3d()
    # holo_gaze()
    holo_cam()
    # tar_holo('gaze')
    
    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'train-v1_2.txt')) as f:
    #     train_list = f.read().splitlines()
    # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'val-v1_2.txt')) as f:
    #     val_list = f.read().splitlines()
    # # with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist', 'data_split', 'test-v1_2.txt')) as f:
    #     test_list = f.read().splitlines()

    # import re
    
    # files = glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/token/*.npz')
    # names = [os.path.basename(x) for x in files]
    # videoname = [re.sub(r'_\d+\.npz', '', name) for name in names]
    # for idx, video in enumerate(videoname):
    #     if video in val_list:
    #         os.system(f'mv {files[idx]} /capstor/scratch/cscs/lgen/datasets_aligned/val/rgb/holoassist/{names[idx]}')

    # tar_holo('rgb')
    # high_res_videos = sorted(glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/*.mp4'))
    # p = Pool(100)
    # p.map_async(downsample_video, high_res_videos)
    # p.close()
    # p.join()
    # print('done')
    # downsample_video('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/z036-june-23-22-gopro_35.mp4')

    # tar original files
    # tar_files('/capstor/scratch/cscs/lgen/datasets_aligned/depth/holoassist')
    # tar_files_hot3d()
    # process_tar_files(tar_dir='/capstor/scratch/cscs/lgen/datasets_aligned/rgb/holoassist/video_tar', output_dir='/capstor/scratch/cscs/lgen/datasets_aligned/depth/')
    # tar_files_hot3d()
    # tar_files_taco()
    # tar_files_h2o()
    # tar_files_arctic()
