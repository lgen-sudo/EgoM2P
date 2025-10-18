import argparse
import numpy as np
import cv2
import os
import tqdm
import sys
import glob

axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


def read_pose_txt(img_pose_path):
    img_pose_array = []
    with open(img_pose_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            # pose = np.array(line_data[1:]).reshape(4, 4)
            # pose = np.dot(axis_transform,pose)
            # line_data[1:] = pose.reshape(-1)
            # print("line_data",line_data)
            # line_data = line.strip().split('\t')
            img_pose_array.append(line_data)
        img_pose_array = np.array(img_pose_array)
    return img_pose_array

def read_intrinsics_txt(img_instrics_path):
    with open(img_instrics_path) as f:
        data = list(map(float, f.read().split('\t')))
        intrinsics = np.array(data[:9]).reshape(3, 3)
        width = data[-2]
        height = data[-1]
    return intrinsics, width, height

def read_gaze_txt(gaze_path):
    with open(gaze_path) as f:
        gaze_data = []
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            gaze_data.append(line_data)
        gaze_data = np.array(gaze_data)
    return gaze_data

def get_eye_gaze_point(gaze_data, dist):
    origin_homog = gaze_data[2:5]
    direction_homog = gaze_data[5:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    point = origin_homog + direction_homog * dist

    return point[:3]


def main(video_name, folder_path="/capstor/store/cscs/swissai/a03/datasets/holoassist"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eye_dist', type=float, default=0.5,
                        help='Eyegaze projection dist is 50cm by default')
    args = parser.parse_args()
    
    base_path = os.path.join(folder_path, video_name, "Export_py")

    if not os.path.exists(base_path):
        # Exit if the path does not exist
        print('{} does not exist'.format(base_path))
        return
        
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
        
    gaze_path = os.path.join(base_path, "Eyes", "Eyes_sync.txt")
    gaze_array = read_gaze_txt(gaze_path)
    gaze_timestamp = gaze_array[:, :2]
    eyeproj_list = []
    
    num_frames = len(img_sync_timing_array)
    # Read campose
    img_pose_path = os.path.join(img_path, 'Pose_sync.txt')
    img_pose_array = read_pose_txt(img_pose_path)
    # Read cam instrics
    img_instrics_path = os.path.join(img_path, 'Intrinsics.txt')
    img_intrinsics, width, height = read_intrinsics_txt(img_instrics_path)
    

    for frame in range(num_frames):
        img_pose = img_pose_array[frame][2:].reshape(4, 4)

        # Put an empty camera pose for image.
        rvec = np.array([[0.0, 0.0, 0.0]])
        tvec = np.array([0.0, 0.0, 0.0])

        gaze_indices = [[frame]]
        point = get_eye_gaze_point(gaze_array[gaze_indices[0][0]], args.eye_dist)
        
        point_transformed = np.dot(axis_transform, np.dot(np.linalg.inv(
            img_pose), np.concatenate((point, [1]))))

        img_points_gaze, _ = cv2.projectPoints(
            point_transformed[:3].reshape((1, 3)), rvec, tvec, img_intrinsics, np.array([]))
        eyeproj_list.append(img_points_gaze[0][0])

    with open(os.path.join(base_path, "Eyes",'Eyes_proj.txt'), 'w') as f:
        for ii, elems in enumerate(eyeproj_list):
            #print("gaze_timestamp",np.shape(gaze_timestamp))
            
            f.write(f"{gaze_timestamp[ii,0]}\t {gaze_timestamp[ii,1]}\t")
            for elem in elems:
                f.write(f"{elem}\t")
            f.write("\n")
        print('save ', video_name)

if __name__ == '__main__':
    for video_name in tqdm.tqdm(glob.glob('/capstor/store/cscs/swissai/a03/datasets/holoassist/*')):
        main(video_name)