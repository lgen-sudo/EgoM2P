import os
import glob
import numpy as np
from tqdm import tqdm
import pdb
import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def is_neighbor(file1, file2):
    # Check if two files are neighbors based on their filename numbers
    parts1 = file1.split('_')
    parts2 = file2.split('_')
    return parts1[:-1] == parts2[:-1] and int(parts1[-1].split('.')[0]) - int(parts2[-1].split('.')[0]) == -1

def temporal_overlap_augmentation(data, stride=10, sequence_length=60):
    augmented_data = []
    for start in range(0, len(data) - sequence_length + 1, stride):
        augmented_data.append(data[start:start + sequence_length])
    return augmented_data

class DatasetLoader:
    def __init__(self, mode):
        self.dataset_samples = []
        self.mode = mode
        self.sample_counts = {}  # To store the number of samples extracted from each dataset
        self.clip_len = 60

        if mode == 'train':
            self._load_train_data()
        elif mode == 'val':
            self._load_val_data()
    
    def read_pose(self, data_path):
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
    
    def canonicalize(self, sample):
        # sample: [60, 4, 4]
        # output: canonicalized 9d camera
        inv = np.linalg.inv(sample[0])
        canoed = np.einsum('ij, kjl -> kil', inv, sample)
        rot6d = canoed[:, :3, :2]
        transl = canoed[:, :3, 3:]
        cam_9d = np.concatenate((rot6d, transl), axis=-1).transpose(0, 2, 1).reshape(-1, 9)
        return cam_9d

    def _load_train_data(self):
        
        # holoassist
        print('holo')
        num = 0
        holocam2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist/', 'data_split', 'train-v1_2.txt')) as f:
            train_list = f.read().splitlines()
        for x in tqdm(train_list):
            data_path = os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist/', x, 'Export_py', 'Video', 'Pose_sync.txt')
            cam_traj = self.read_pose(data_path)
            for kk in range(0, cam_traj.shape[0] - self.clip_len + 1, 10):
                # self.dataset_samples.append(self.canonicalize(cam_traj[kk : kk + self.clip_len]))
                self.dataset_samples.append(self.canonicalize(cam_traj[kk : kk + self.clip_len] @ holocam2opencv))
                num += 1
                # if num == 282766:
                #     pdb.set_trace()
        self.sample_counts['holo_train'] = len(self.dataset_samples)
        print(num)
        # pdb.set_trace()
        np.save('holo_cam_opencv.npy', self.dataset_samples)
        
        # # HOT3D
        print('hot3d')
        self.dataset_samples = []
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/train_files.txt', 'r') as f:
            train_list = f.read().splitlines()
        train_list = sorted(train_list, key=natural_sort_key)
        self.sample_counts['hot3d_train'] = len(train_list)
        # for x in tqdm(train_list):
        #     npz_name = x.split('.')[0]
        #     cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d', f'{npz_name}.npz'))['cam']
        #     self.dataset_samples.append(self.canonicalize(cam_traj))
        for i in tqdm(range(len(train_list) - 1)):
            npz_name1 = train_list[i].split('.')[0]
            npz_name2 = train_list[i + 1].split('.')[0]

            if not is_neighbor(npz_name1, npz_name2):
                # print(npz_name1, ' ', npz_name2, ' not neighbor')
                continue

            cam_traj1 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d', f'{npz_name1}.npz'))['cam']
            cam_traj2 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d', f'{npz_name2}.npz'))['cam']

            if np.any(np.isnan(cam_traj1)) or np.any(np.isnan(cam_traj2)):
                continue

            combined_traj = np.concatenate((cam_traj1, cam_traj2), axis=0)
            augmented_sequences = temporal_overlap_augmentation(combined_traj)

            for seq in augmented_sequences:
                self.dataset_samples.append(self.canonicalize(seq))
        np.save('hot3d_cam.npy', self.dataset_samples)
        
        # # ARCTIC
        print('arctic')
        self.dataset_samples = []
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/arctic/train_files.txt', 'r') as f:
            train_list = f.read().splitlines()
        train_list = sorted(train_list, key=natural_sort_key)
        self.sample_counts['arctic_train'] = len(train_list)
        # for x in tqdm(train_list):
        #     npz_name = x.split('.')[0]
        #     cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/arctic', f'{npz_name}.npy'))
        #     self.dataset_samples.append(self.canonicalize(cam_traj))
        for i in tqdm(range(len(train_list) - 1)):
            npz_name1 = train_list[i].split('.')[0]
            npz_name2 = train_list[i + 1].split('.')[0]

            if not is_neighbor(npz_name1, npz_name2):
                # print(npz_name1, ' ', npz_name2, ' not neighbor')
                continue

            cam_traj1 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/arctic', f'{npz_name1}.npy'))
            cam_traj2 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/arctic', f'{npz_name2}.npy'))

            if np.any(np.isnan(cam_traj1)) or np.any(np.isnan(cam_traj2)):
                continue

            combined_traj = np.concatenate((cam_traj1, cam_traj2), axis=0)
            augmented_sequences = temporal_overlap_augmentation(combined_traj)

            for seq in augmented_sequences:
                self.dataset_samples.append(self.canonicalize(seq))
        np.save('arctic_cam.npy', self.dataset_samples)

        # # TACO
        print('taco')
        self.dataset_samples = []
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/taco/train_files.txt', 'r') as f:
            train_list = f.read().splitlines()
        train_list = sorted(train_list, key=natural_sort_key)
        self.sample_counts['taco_train'] = len(train_list)
        for x in tqdm(train_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/taco', f'{npz_name}.npy'))
            cam_aug = temporal_overlap_augmentation(cam_traj)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
    
        # # H2O
        print('h2o')
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/h2o/train_files.txt', 'r') as f:
            train_list = f.read().splitlines()
        train_list = sorted(train_list, key=natural_sort_key)
        self.sample_counts['h2o_train'] = len(train_list)
        for x in tqdm(train_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/h2o', f'{npz_name}.npy'))
            cam_aug = temporal_overlap_augmentation(cam_traj)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
        np.save('taco_h2o_cam.npy', self.dataset_samples)

        # EgoExo4D. the original camera pose is x up, y right, z forward. needs to convert to opencv
        print('egoexo')
        cw90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.dataset_samples = []
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/train_samples.txt', 'r') as f:
            train_list = f.read().splitlines()
        train_list = sorted(train_list, key=natural_sort_key)
        self.sample_counts['egoexo4d_train'] = len(train_list)
        # for x in tqdm(train_list):
        #     npz_name = x.split('.')[0]
        #     cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label', f'{npz_name}.npz'))['cam']
        #     if np.any(np.isnan(cam_traj)):
        #         continue
        #     self.dataset_samples.append(self.canonicalize(cam_traj))
        
        for i in tqdm(range(len(train_list) - 1)):
            npz_name1 = train_list[i].split('.')[0]
            npz_name2 = train_list[i + 1].split('.')[0]

            if not is_neighbor(npz_name1, npz_name2):
                # print(npz_name1, ' ', npz_name2, ' not neighbor')
                continue

            cam_traj1 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label', f'{npz_name1}.npz'))['cam']
            cam_traj2 = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label', f'{npz_name2}.npz'))['cam']

            if np.any(np.isnan(cam_traj1)) or np.any(np.isnan(cam_traj2)):
                continue

            combined_traj = np.concatenate((cam_traj1, cam_traj2), axis=0)
            augmented_sequences = temporal_overlap_augmentation(combined_traj)

            for seq in augmented_sequences:
                # self.dataset_samples.append(self.canonicalize(seq)) # original camera convention
                # convert to opencv camera:
                self.dataset_samples.append(self.canonicalize(seq @ cw90))
        np.save('egoexo_cam_opencv.npy', self.dataset_samples)

        # Egogen
        print('egogen')
        self.dataset_samples = []
        opengl_to_opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        egogen_files = glob.glob('/iopsstor/scratch/cscs/lgen/egogen_new/cam/*.npz')
        # egogen_files = sorted(egogen_files, key=natural_sort_key)
        self.sample_counts['egogen_train'] = len(egogen_files)
        for file in tqdm(egogen_files):
            cam_traj = np.load(file)['arr_0']
            cam_aug = temporal_overlap_augmentation(cam_traj)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam @ opengl_to_opencv)) # opengl to opencv
        
        np.save('egogen_cam_opencv.npy', self.dataset_samples)
                

        # all_data = np.concatenate(self.dataset_samples)
        # self.mean = all_data.mean(0)
        # self.std = all_data.std(0)
        
        # print('mean', self.mean)
        # print('std', self.std)
        
        # np.save('cam_mean_overlap_more.npy', self.mean)
        # np.save('cam_std_overlap_more.npy', self.std)
        # np.save('cam_train_overlap_more.npy', self.dataset_samples)
        pass

    def _load_val_data(self):
        # HoloAssist
        print('holo')
        holocam2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        # num = 0
        with open(os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist/', 'data_split', 'val-v1_2.txt')) as f:
            val_list = f.read().splitlines()
        for x in tqdm(val_list):
            data_path = os.path.join('/capstor/store/cscs/swissai/a03/datasets/holoassist/', x, 'Export_py', 'Video', 'Pose_sync.txt')
            cam_traj = self.read_pose(data_path)
            for kk in range(0, cam_traj.shape[0] - self.clip_len + 1, 10):
                self.dataset_samples.append(self.canonicalize(cam_traj[kk : kk + self.clip_len] @ holocam2opencv))
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()
        self.sample_counts['holo_val'] = len(self.dataset_samples)
        # np.save('holo_cam_opencv_val.npy', self.dataset_samples)

        # # HOT3D
        print('hot3d')
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/val_files.txt', 'r') as f:
            val_list = f.read().splitlines()
        self.sample_counts['hot3d_val'] = 0
        # for x in val_list:
        #     self.dataset_samples.append(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/rgb_256_8fps', x))
        for x in tqdm(val_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d', f'{npz_name}.npz'))['cam']
            cam_aug = temporal_overlap_augmentation(cam_traj, stride=30)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
                self.sample_counts['hot3d_val'] += 1
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()
        
        # # ARCTIC
        print('arctic')
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/arctic/val_files.txt', 'r') as f:
            val_list = f.read().splitlines()
        self.sample_counts['arctic_val'] = 0
        # for x in val_list:
        #     self.dataset_samples.append(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/arctic/rgb_256_8fps', x))
        for x in tqdm(val_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/arctic', f'{npz_name}.npy'))
            cam_aug = temporal_overlap_augmentation(cam_traj, stride=30)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
                self.sample_counts['arctic_val'] += 1
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()

        # # TACO
        print('taco')
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/taco/val_files.txt', 'r') as f:
            val_list = f.read().splitlines()
        self.sample_counts['taco_val'] = 0
        # for x in val_list:
        #     self.dataset_samples.append(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/taco/rgb_256_8fps', x))
        for x in tqdm(val_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/taco', f'{npz_name}.npy'))
            # self.dataset_samples.append(self.canonicalize(cam_traj))
            cam_aug = temporal_overlap_augmentation(cam_traj, stride=30)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
                self.sample_counts['taco_val'] += 1
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()

        # # H2O
        print('h2o')
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/h2o/val_files.txt', 'r') as f:
            val_list = f.read().splitlines()
        self.sample_counts['h2o_val'] = 0
        # for x in val_list:
        #     self.dataset_samples.append(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/h2o/rgb_256_8fps', x))
        for x in tqdm(val_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/cam/h2o', f'{npz_name}.npy'))
            cam_aug = temporal_overlap_augmentation(cam_traj, stride=30)
            for cam in cam_aug:
                self.dataset_samples.append(self.canonicalize(cam))
                self.sample_counts['h2o_val'] += 1
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()

        # # EgoExo4D
        print('egoexo')
        cw90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/val_samples.txt', 'r') as f:
            val_list = f.read().splitlines()
        self.sample_counts['egoexo4d_val'] = 0
        # for x in val_list:
        #     self.dataset_samples.append(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/rgb_256_8fps', x))
        for x in tqdm(val_list):
            npz_name = x.split('.')[0]
            cam_traj = np.load(os.path.join('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label', f'{npz_name}.npz'))['cam']
            if np.any(np.isnan(cam_traj)):
                continue
            # self.dataset_samples.append(self.canonicalize(cam_traj))
            cam_aug = temporal_overlap_augmentation(cam_traj, stride=30)
            for cam in cam_aug:
                # pdb.set_trace()
                self.dataset_samples.append(self.canonicalize(cam @ cw90))
                self.sample_counts['egoexo4d_val'] += 1
                # num += 1
                # if num >= 164260:
                #     pdb.set_trace()
        
        np.save('/iopsstor/scratch/cscs/lgen/cam_tok_train_data/60frames_opencv/cam_opencv_val.npy', self.dataset_samples)
        pass

    def get_sample_counts(self):
        return self.sample_counts

# Example usage
mode = 'val'  # or 'val'
loader = DatasetLoader(mode)
sample_counts = loader.get_sample_counts()
print("Sample counts:", sample_counts)