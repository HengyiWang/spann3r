import os
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Scannetpp(BaseStereoViewDataset):
    def __init__(self, num_frames=5, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames

        # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scenes) * self.num_seq
        
    def load_all_scenes(self, base_dir, num_seq=200):
        
        meta_split = osp.join(base_dir, 'splits', f'nvs_sem_{self.split}')
        
        if not osp.exists(meta_split):
            raise FileNotFoundError(f"Split file {meta_split} not found")
        
        with open(meta_split) as f:
            scene_list = f.read().splitlines()
            
        print(f"Found {len(scene_list)} scenes in split {self.split}")
        
        
        
        
        
        
        self.scenes = {}
        
        data_all = os.listdir(base_dir)
        print('All datasets in Habitat:', data_all)
        
        for data in data_all:
            scenes = os.listdir(osp.join(base_dir, data))
            self.scenes[data] = scenes
        
        self.scenes = {(k, v2): list(range(num_seq)) for k, v in self.scenes.items() 
                           for v2 in v}
        self.scene_list = list(self.scenes.keys())


        
        
            
            
            
            
            
        
        
        
        
        
        
        
            
        
        
        


if __name__ == '__main__':
    dataset = Scannetpp(split='train', ROOT="/media/hengyi/Data/scannet++", resolution=224)
    
    

        
        
    