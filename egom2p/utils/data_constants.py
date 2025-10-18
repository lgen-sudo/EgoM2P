# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

# HOLOASSIST_MEAN = (0.14020734,  0.05260414, -0.73053254, -0.08709813,  0.2147107, 0.0261649 , -0.3997017 ,  0.21050708,  0.00756409)
# HOLOASSIST_STD = (0.47595015,  0.41394154,  0.21453851,  0.62474589,  0.72739242, 0.16194923, 11.04588997,  8.13007173,  1.55225898)

IMAGENET_SURFACE_NORMAL_MEAN = (0.501, 0.405, 0.137)
IMAGENET_SURFACE_NORMAL_STD = (0.114, 0.165, 0.081)

SEG_IGNORE_INDEX = 255
SEG_IGNORE_INDEX_V2 = 0
PAD_MASK_VALUE = 254
COCO_SEMSEG_NUM_CLASSES = 133 + 1  # One extra class for no-class
ADE20K_SEMSEG_NUM_CLASSES = 150 + 1  # One extra class for no-class
HYPERSIM_SEMSEG_NUM_CLASSES = 41


IMAGE_TASKS = {'rgb', 'depth', 'semseg', 'semseg_hypersim', 'semseg_coco', 'semseg_ade20k', 'normal'}
DETECTION_TASKS = {'det'} # 'det_coco', 'det_lvis'
TEXT_TASKS = {'caption'}
VISION_TASKS = IMAGE_TASKS | DETECTION_TASKS
SEQUENCE_TASKS = DETECTION_TASKS | TEXT_TASKS

NYU_MEAN = 2070.7764
NYU_STD = 777.5723

# these values are computed using camera traj tokenizer training set produced by egom2p/data/load_cam.py
CAM_MEAN = [0.9330503976511908, 0.004411310479848365, -0.004745505205447648, -0.004031105018382311, 0.9670221745404669, 0.01229744056849661, 0.0025366981809852238, -0.002534521133857395, 0.03687088469884694]
CAM_STD = [0.22410282941424692, 0.15409594854789294, 0.23538762722445528, 0.151592393482513  , 0.09890854498671052, 0.17871060521456542, 0.16229749698697332, 0.11305994570310635, 0.22450155613920986]