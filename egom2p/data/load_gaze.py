import os
import glob
import tqdm
import numpy as np
import pdb

class DatasetLoader:
    def __init__(self, mode):
        self.dataset_samples = []
        self.mode = mode
        self.sample_counts = {}  # To store the number of samples extracted from each dataset
        self.clip_len = 60
        self.data_path = '/capstor/store/cscs/swissai/a03/datasets/holoassist'
        
        self.mean = (0.5, 0.5) # [0, 1] -> [-1, 1]
        self.std = (0.5, 0.5)

        if mode == 'train':
            self._load_train_data()
        elif mode == 'val':
            self._load_val_data()
    
    def read_gaze_txt(self, gaze_path):
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
    
    def convert(self, gaze_data, orig_res, resize_res, new_res=[480, 480]):
        # convert gaze 2d coordinates in the original resolution to 480x480
        orig_res = np.array(orig_res)
        new_res = np.array(new_res)
        gaze_normed = gaze_data / orig_res # to [0, 1]
        gaze_resize_coord = gaze_normed * np.array(resize_res) # resized coord

        # check if valid in center cropped image (new_res)
        _min = (resize_res - new_res) / 2
        gaze_new_coord = gaze_resize_coord - _min
        gaze = gaze_new_coord / np.array(new_res)

        mask = np.ones(gaze.shape[0]) # invalid val: 0
        nan = np.where(np.isnan(gaze).any(-1))[0]
        mask[nan] = 0
        gaze[nan] = 0. # inpute nan with 0.

        # many noise in the GT data, filter out gaze very outside of the image
        out = np.where((gaze > 1.2).any(-1))[0]
        mask[out] = 0
        gaze[out] = 0.
        out = np.where((gaze < -0.2).any(-1))[0]
        mask[out] = 0
        gaze[out] = 0

        gaze = (gaze - self.mean) / self.std # normalize data to -1,1
        return np.concatenate([gaze, mask.reshape(-1, 1)], axis=-1)

    def _load_train_data(self):
        # HoloAssist
        self.sample_counts['holo_train'] = 0
        with open(os.path.join(self.data_path, 'data_split', 'train-v1_2.txt')) as f:
            train_list = f.read().splitlines()
        for x in tqdm.tqdm(train_list):
            data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
            gaze_data = self.read_gaze_txt(data_path)
            for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
                self.dataset_samples.append(self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504]))
                self.sample_counts['holo_train'] += 1

        # EgoExo4D
        self.sample_counts['egoexo_train'] = 0
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/train_samples.txt') as f:
            train_list = f.read().splitlines()
        for mp4_name in tqdm.tqdm(train_list):
            npz_name = mp4_name.split('.')[0]
            gaze_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label/{npz_name}.npz')['gaze']
            # self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
            converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
            self.dataset_samples.append(converted[:self.clip_len])
            self.dataset_samples.append(converted[self.clip_len:])
            self.sample_counts['egoexo_train'] += 2

        # HOT3D
        self.sample_counts['hot3d_train'] = 0
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/train_files.txt') as f:
            train_list = f.read().splitlines()
        for mp4_name in tqdm.tqdm(train_list):
            npz_name = mp4_name.split('.')[0]
            all_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/{npz_name}.npz')
            if 'gaze' in all_data.files:
                gaze_data = all_data['gaze']
            else:
                continue
            # self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
            converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
            self.dataset_samples.append(converted[:self.clip_len])
            self.dataset_samples.append(converted[self.clip_len:])
            self.sample_counts['hot3d_train'] += 2
        
        np.save('gaze_train_60.npy', self.dataset_samples)


    def _load_val_data(self):
        # HoloAssist
        self.sample_counts['holo_val'] = 0
        with open(os.path.join(self.data_path, 'data_split', 'val-v1_2.txt')) as f:
            val_list = f.read().splitlines()
        for x in tqdm.tqdm(val_list):
            data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
            gaze_data = self.read_gaze_txt(data_path)
            for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
                self.dataset_samples.append(self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504]))
                self.sample_counts['holo_val'] += 1

        # EgoExo4D
        self.sample_counts['egoexo_val'] = 0
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/val_samples.txt') as f:
            val_list = f.read().splitlines()
        for mp4_name in tqdm.tqdm(val_list):
            npz_name = mp4_name.split('.')[0]
            gaze_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label/{npz_name}.npz')['gaze']
            # self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
            converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
            self.dataset_samples.append(converted[:self.clip_len])
            self.dataset_samples.append(converted[self.clip_len:])
            self.sample_counts['egoexo_val'] += 2
        
        # HOT3D
        self.sample_counts['hot3d_val'] = 0
        with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/val_files.txt') as f:
            val_list = f.read().splitlines()
        for mp4_name in tqdm.tqdm(val_list):
            npz_name = mp4_name.split('.')[0]
            all_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/{npz_name}.npz')
            if 'gaze' in all_data.files:
                gaze_data = all_data['gaze']
            else:
                continue
            converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
            self.dataset_samples.append(converted[:self.clip_len])
            self.dataset_samples.append(converted[self.clip_len:])
            self.sample_counts['hot3d_val'] += 2
        
        np.save('gaze_val_60.npy', self.dataset_samples)

    def get_sample_counts(self):
        return self.sample_counts

# Example usage
mode = 'train'  # or 'val'
loader = DatasetLoader(mode)
sample_counts = loader.get_sample_counts()
print("Sample counts:", sample_counts)
