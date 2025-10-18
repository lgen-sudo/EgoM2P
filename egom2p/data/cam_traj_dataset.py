import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pdb
import glob
import tarfile
import io
from egom2p.utils.data_constants import CAM_MEAN, CAM_STD

class CamTrajDataset(Dataset):
    # canonicalize camera trajectory with the first pose
    def __init__(self, mode='train', clip_len=60, args=None):
        
        self.data_path = args.data_path
        # self.data_path = "/capstor/store/cscs/swissai/a03/datasets/holoassist" 
        self.mode = mode
        self.clip_len = clip_len # num of frames. cut videos into clips
        assert self.clip_len == 60 # changed to 60 frames 2s.
        self.args = args
        
        self.dataset_samples = []
        # self.all_data = []
        # num = 0
        self.mean = np.array(CAM_MEAN)
        self.std = np.array(CAM_STD)
        
        if mode == 'train':
            # with open(os.path.join(self.data_path, 'data_split', 'train-v1_2.txt')) as f:
            #     train_list = f.read().splitlines()
            # for x in train_list:
            #     data_path = os.path.join(self.data_path, x, 'Export_py', 'Video', 'Pose_sync.txt')
            #     cam_traj = self.read_pose(data_path)
            #     # TODO: change this coord system to aria world coord is not needed due to canonicalization
            #     for kk in range(0, cam_traj.shape[0] - self.clip_len + 1, 10):
            #         self.dataset_samples.append(self.canonicalize(cam_traj[kk : kk + self.clip_len]))

            # all_data = np.concatenate(self.dataset_samples)
            # self.mean = all_data.mean(0)
            # self.std = all_data.std(0)
            # self.mean = np.load()
            # self.std = np.load()
            
            # change to your path, processed by load_cam.py
            self.dataset_samples = np.load('/iopsstor/scratch/cscs/lgen/cam_tok_train_data/60frames_opencv/all_cam_opencv_train.npy')
            print('load 60frames_opencv/all_cam_opencv_train.npy')
        elif mode == 'val':
            # with open(os.path.join(self.data_path, 'data_split', 'val-v1_2.txt')) as f:
            #     val_list = f.read().splitlines()
            # for x in val_list:
            #     data_path = os.path.join(self.data_path, x, 'Export_py', 'Video', 'Pose_sync.txt')
            #     cam_traj = self.read_pose(data_path)
            #     for kk in range(0, cam_traj.shape[0] - self.clip_len + 1, 10):
            #         self.dataset_samples.append(self.canonicalize(cam_traj[kk : kk + self.clip_len]))
            
            # change to your path, processed by load_cam.py
            self.dataset_samples = np.load('/iopsstor/scratch/cscs/lgen/cam_tok_train_data/60frames_opencv/cam_opencv_val.npy')
            print('load 60frames_opencv/cam_opencv_val.npy')
        elif mode == 'tokenize':
            if 'example' in args.tokenize_path:
                # TODO: ensure the input camera pose is in opencv format with shape 60x4x4!
                cam_data = np.load(args.tokenize_path)
                self.dataset_samples.append({'x': self.canonicalize(cam_data), 'name': os.path.basename(args.tokenize_path).split('.')[0] + '-tok' if not args.recon else os.path.basename(args.tokenize_path).split('.')[0] + '-recon'})
            elif '/iopsstor/scratch/cscs/cyutong/egodata/egoexo/label' in args.tokenize_path and args.recon:
                cw90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                for file in glob.glob(os.path.join(args.tokenize_path, '*')):
                    cam_data = np.load(file)['cam']
                    self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len] @ cw90), 'name': os.path.basename(file).split('.')[0] + '-0'})
            elif '/iopsstor/scratch/cscs/lgen/eval_result/adt_cam/fisheye/cam_gt' in args.tokenize_path and args.recon:
                for file in glob.glob(os.path.join(args.tokenize_path, '*')):
                    cam_data = np.load(file)
                    self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len]), 'name': os.path.basename(file).split('.')[0]})
            elif 'egoexo' in args.tokenize_path:
                cw90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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
                                    # Check if 'cam' is in the .npz file
                                    if 'cam' in npz_file:
                                        # Access the 'cam' data
                                        cam_data = npz_file['cam']
                                        if np.any(np.isnan(cam_data)):
                                            continue
                                        self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len] @ cw90), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                        self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:] @ cw90), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
                                        
            elif 'egogen' in args.tokenize_path:
                opengl_to_opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                for file in glob.glob('/iopsstor/scratch/cscs/lgen/egogen_new/cam/*.npz'):
                    cam_traj = np.load(file)['arr_0']
                    self.dataset_samples.append({'x': self.canonicalize(cam_traj[:self.clip_len] @ opengl_to_opencv), 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': self.canonicalize(cam_traj[self.clip_len:] @ opengl_to_opencv), 'name': os.path.basename(file).split('.')[0] + '-1'})
            # for file in glob.glob(os.path.join(args.tokenize_path, '*.npy')):
            #     self.dataset_samples.append(file)
                # name = os.path.basename(file).split('.')[0]
                # cam_data = np.load(file)
                # self.dataset_samples.append({'x': self.canonicalize(cam_data), 'name': name})
            elif 'h2o.tar' in args.tokenize_path or 'taco.tar' in args.tokenize_path or 'arctic.tar' in args.tokenize_path:
                with tarfile.open(args.tokenize_path, 'r') as tar:
                    # Iterate over each file in the tar archive
                    for member in tar.getmembers():
                        # Check if the file is a .npz file within the 'label/' directory
                        if member.isfile() and member.name.endswith('.npy'):
                            # Extract the file into memory
                            file_obj = tar.extractfile(member)
                            
                            if file_obj is not None:
                                cam_data = np.load(io.BytesIO(file_obj.read()))
                                if np.any(np.isnan(cam_data)):
                                    continue
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len]), 'name': os.path.basename(member.name).split('.')[0] + '-0'})
                                self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:]), 'name': os.path.basename(member.name).split('.')[0] + '-1'})
            elif 'holoassist' in args.tokenize_path:
                holocam2opencv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                # for file in glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/cam/holoassist/*.npy'):
                for file in glob.glob('/iopsstor/scratch/cscs/lgen/tmp/cam/holoassist/*.npy'):
                    cam_traj = np.load(file)
                    self.dataset_samples.append({'x': self.canonicalize(cam_traj[:self.clip_len] @ holocam2opencv), 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': self.canonicalize(cam_traj[self.clip_len:] @ holocam2opencv), 'name': os.path.basename(file).split('.')[0] + '-1'})
            
            elif 'hot3d' in args.tokenize_path:
                for file in glob.glob('/capstor/scratch/cscs/lgen/datasets_aligned/hot3d/*.npz'):
                    cam_data = np.load(file)['cam']
                    if np.isnan(np.any(cam_data)):
                        continue
                    self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len]), 'name': os.path.basename(file).split('.')[0] + '-0'})
                    self.dataset_samples.append({'x': self.canonicalize(cam_data[self.clip_len:]), 'name': os.path.basename(file).split('.')[0] + '-1'})
                
            elif 'adt' in args.tokenize_path:
                for file in glob.glob(f'{args.tokenize_path}/*.npz'):
                    cam_data = np.load(file)['cam']
                    if np.isnan(np.any(cam_data)):
                        continue
                    self.dataset_samples.append({'x': self.canonicalize(cam_data[:self.clip_len]), 'name': os.path.basename(file).split('.')[0]})
    
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

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = self.dataset_samples[index]
            return torch.tensor((sample - self.mean) / self.std).float()
        elif self.mode == 'val':
            sample = self.dataset_samples[index]
            if (self.mean == 0).all() or (self.std == 1).all(): # mean should be set in run_.py
                pdb.set_trace()
            return torch.tensor((sample - self.mean) / self.std).float()
        elif self.mode == 'tokenize':
            # data_path = self.dataset_samples[index]
            # sample = self.canonicalize(np.load(data_path))
            # name = os.path.basename(data_path).split('.')[0]
            if (self.mean == 0).all() or (self.std == 1).all():
                pdb.set_trace()
            data = self.dataset_samples[index]
            return {'x': torch.tensor((data['x'] - self.mean) / self.std).float(), 'name': data['name']}
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

if __name__ == '__main__':
    x = CamTrajDataset()
    pdb.set_trace()
