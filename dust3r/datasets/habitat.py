import os
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class habitat(BaseStereoViewDataset):
    def __init__(self, num_seq=200, num_frames=5, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames

        # load all scenes
        self.load_all_scenes(ROOT, num_seq)
    
    def __len__(self):
        return len(self.scenes) * self.num_seq
        
    def load_all_scenes(self, base_dir, num_seq=200):
        
        self.scenes = {}
        
        data_all = os.listdir(base_dir)
        print('All datasets in Habitat:', data_all)
        
        for data in data_all:
            scenes = os.listdir(osp.join(base_dir, data))
            self.scenes[data] = scenes
        
        self.scenes = {(k, v2): list(range(num_seq)) for k, v in self.scenes.items() 
                           for v2 in v}
        self.scene_list = list(self.scenes.keys())
        
    
    def _get_views(self, idx, resolution, rng): 
        data, scene = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq
        
        views = []
        imgs_idxs = deque(range(1, self.num_frames+1))
        # TODO: add a bit of randomness of the order
        
        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}.jpeg")
            depthpath = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}_depth.exr")
            cam_params_path = osp.join(self.ROOT, data, scene, f"{seq_id:08}_{im_idx}_camera_params.json")
            
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            # check nan in depth, throw a warning
            if np.isnan(depthmap).any():
                print(f'Warning: NaN in depthmap: {depthpath}, converting to 0.0')
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
            
            cam_params = json.load(open(cam_params_path, 'r'))
            intrinsics = np.array(cam_params['camera_intrinsics'])
            
            # cam_r: [3, 3], cam_t: [3, ]
            cam_r = np.array(cam_params['R_cam2world'], dtype=np.float32)
            cam_t = np.array(cam_params['t_cam2world'], dtype=np.float32)
            
            # camera_pose: [4, 4]
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = cam_r
            camera_pose[:3, 3] = cam_t
            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                continue
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='habitat',
                label=osp.join(data, scene),
                instance=osp.split(impath)[1],
            ))
        return views
            
            

        
        
            
            
            
            
            
        
        
        
        
        
        
        
            
        
        
        


if __name__ == '__main__':
    dataset = habitat(split='train', ROOT="/home/hengyi/nopemap/data/pair_5_subset", resolution=224)
    views = dataset._get_views(0, [256, 256], np.random.RandomState(0))
    
    print(views[0]['instance'])
    
    

        
        
    