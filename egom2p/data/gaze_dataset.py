import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pdb
import glob
import tarfile
import io

class GazeDataset(Dataset):
    def __init__(self, mode='train', clip_len=60, args=None):

        self.data_path = args.data_path
        # self.data_path = "/capstor/store/cscs/swissai/a03/datasets/holoassist" 
        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        self.args = args
        
        self.dataset_samples = []
        self.mean = (0.5, 0.5) # [0, 1] -> [-1, 1]
        self.std = (0.5, 0.5)

        if mode == 'train':
            # load npy data directly. data processing is same as following code.
            # change to your path, processed by load_gaze.py
            self.dataset_samples = np.load('/capstor/scratch/cscs/lgen/datasets_aligned/gaze/all_data/gaze_train_60.npy')

            # with open(os.path.join(self.data_path, 'data_split', 'train-v1_2.txt')) as f:
            #     train_list = f.read().splitlines()
            # for x in train_list:
            #     data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
            #     gaze_data = self.read_gaze_txt(data_path)
            #     for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
            #         self.dataset_samples.append(self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504]))

            # # egoexo4d
            # with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/train_samples.txt') as f:
            #     train_list = f.read().splitlines()
            # for mp4_name in train_list:
            #     npz_name = mp4_name.split('.')[0]
            #     gaze_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label/{npz_name}.npz')['gaze']
            #     self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))

            # # HOT3D
            # with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/train_files.txt') as f:
            #     train_list = f.read().splitlines()
            # for mp4_name in train_list:
            #     npz_name = mp4_name.split('.')[0]
            #     all_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/{npz_name}.npz')
            #     if 'gaze' in all_data.files:
            #         gaze_data = all_data['gaze']
            #     else:
            #         continue
            #     self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))

        elif mode == 'val':
            # load npy data directly. data processing is same as following code.
            # change to your path, processed by load_gaze.py
            self.dataset_samples = np.load('/capstor/scratch/cscs/lgen/datasets_aligned/gaze/all_data/gaze_val_60.npy')

            # with open(os.path.join(self.data_path, 'data_split', 'val-v1_2.txt')) as f:
            #     val_list = f.read().splitlines()
            # for x in val_list:
            #     data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
            #     gaze_data = self.read_gaze_txt(data_path)
            #     for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
            #         self.dataset_samples.append(self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504]))
            
            # # egoexo4d
            # with open('/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/val_samples.txt') as f:
            #     val_list = f.read().splitlines()
            # for mp4_name in val_list:
            #     npz_name = mp4_name.split('.')[0]
            #     gaze_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/egoexo/label/{npz_name}.npz')['gaze']
            #     self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
            
            # # HOT3D
            # with open('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/val_files.txt') as f:
            #     val_list = f.read().splitlines()
            # for mp4_name in val_list:
            #     npz_name = mp4_name.split('.')[0]
            #     all_data = np.load(f'/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/{npz_name}.npz')
            #     if 'gaze' in all_data.files:
            #         gaze_data = all_data['gaze']
            #     else:
            #         continue
            #     self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
        
        elif mode == 'val_plot':
            with open(os.path.join(self.data_path, 'data_split', 'val-v1_2.txt')) as f:
                val_list = f.read().splitlines()
            for x in val_list:
                data_path = os.path.join(self.data_path, x, 'Export_py', 'Eyes', 'Eyes_proj.txt')
                gaze_data = self.read_gaze_txt(data_path)
                for kk in range(0, gaze_data.shape[0] - self.clip_len + 1, 10):
                    sample = self.convert(gaze_data[kk : kk + self.clip_len], orig_res=[896, 504], resize_res=[896, 504])
                    self.dataset_samples.append({'vid': os.path.join(self.data_path, x, 'Export_py', 'Video_pitchshift.mp4'), 'x': sample, 'start_frame': kk, 'end_frame': kk + self.clip_len})

        elif mode == 'tokenize':
            # for file in glob.glob(os.path.join(args.tokenize_path, '*.npy')):
            #     # name = os.path.basename(file).split('.')[0]
            #     # gaze_data = np.load(file)
            #     # self.dataset_samples.append({'x': self.convert(gaze_data, orig_res=[896, 504], resize_res=[896, 504]), 'name': name})
            #     self.dataset_samples.append(file)
            if 'example' in args.tokenize_path:
                gaze_data = np.load(args.tokenize_path) # ensure input gaze data is in [0,1] range
                # because this example data is from holoassist dataset, we use the following params to convert it to desired ranges
                self.dataset_samples.append({'x': self.convert(gaze_data, orig_res=[896, 504], resize_res=[896, 504], new_res=[480, 480]), 'name': os.path.basename(args.tokenize_path).split('.')[0] + '-tok' if not args.recon else os.path.basename(args.tokenize_path).split('.')[0] + '-recon'})
            elif '/iopsstor/scratch/cscs/lgen/eval_result/egoexo_gaze/label' in args.tokenize_path and args.recon:
                for file in glob.glob(os.path.join(args.tokenize_path, '*')):
                    gaze_data = np.load(file)['gaze']
                    converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                    self.dataset_samples.append({'x': converted[:self.clip_len], 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': converted[self.clip_len:], 'name': os.path.basename(file).split('.')[0] + '-1'})
                    
            elif 'egoexo' in args.tokenize_path:
                with tarfile.open(os.path.join(args.tokenize_path, 'label.tar'), 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.startswith('label/') and member.name.endswith('.npz'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                # Load the .npz file using numpy
                                with np.load(io.BytesIO(file_obj.read())) as npz_file:
                                    if 'gaze' in npz_file:
                                        gaze_data = npz_file['gaze']
                                        converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                                        self.dataset_samples.append({'x': converted[:self.clip_len], 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                        self.dataset_samples.append({'x': converted[self.clip_len:], 'name': os.path.basename(member.name).split('.')[0] + '-1'})
                                        
            elif 'holoassist' in args.tokenize_path:
                for file in glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/gaze/holoassist/*.npy'):
                    gaze_data = np.load(file)
                    # converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                    converted = self.convert(gaze_data, orig_res=[896, 504], resize_res=[896, 504])
                    self.dataset_samples.append({'x': converted[:self.clip_len], 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': converted[self.clip_len:], 'name': os.path.basename(file).split('.')[0] + '-1'})
            
            elif 'hot3d' in args.tokenize_path:
                for file in glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/*.npz'):
                    all_data = np.load(file)
                    if 'gaze' in all_data.files:
                        gaze_data = all_data['gaze']
                    else:
                        continue
                    # self.dataset_samples.append(self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408]))
                    converted = self.convert(gaze_data, orig_res=[1408, 1408], resize_res=[1408, 1408], new_res=[1408, 1408])
                    self.dataset_samples.append({'x': converted[:self.clip_len], 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': converted[self.clip_len:], 'name': os.path.basename(file).split('.')[0] + '-1'})


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

    def __getitem__(self, index):
        if self.mode == 'val_plot':
            data = self.dataset_samples[index]
            return {'vid': data['vid'], 
                    'x': torch.tensor(data['x']).float(),
                    'start_frame': data['start_frame'],
                    'end_frame': data['end_frame']}
        elif self.mode == 'tokenize':
            # data_path = self.dataset_samples[index]
            # name = os.path.basename(data_path).split('.')[0]
            # gaze_data = self.convert(np.load(data_path), orig_res=[896, 504], resize_res=[896, 504])
            # # self.dataset_samples.append({'x': self.convert(gaze_data, orig_res=[896, 504], resize_res=[896, 504]), 'name': name})
            # return {'x': torch.tensor(gaze_data).float(), 'name': name}
            data = self.dataset_samples[index]
            return {'x':torch.tensor(data['x']).float(), 'name': data['name']}
        else:
            return torch.tensor(self.dataset_samples[index]).float()

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

if __name__ == '__main__':
    x = GazeDataset()
    pdb.set_trace()
