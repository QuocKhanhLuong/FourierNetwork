"""
Synapse Dataset Loader for 2D Training
Loads preprocessed .npy volumes (Multi-organ CT)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import json


class SynapseDataset2D(Dataset):
    """
    Synapse Multi-Organ CT Dataset for 2D slice training.
    
    Args:
        npy_dir: Path to preprocessed data (with volumes/ and masks/ subdirs)
        use_memmap: Use memory-mapped loading
        in_channels: 1 or 3 (replicate single channel)
        max_cache: Max volumes to cache
    """
    
    def __init__(self, npy_dir, use_memmap=True, in_channels=3, max_cache=10):
        self.use_memmap = use_memmap
        self.in_channels = in_channels
        self.max_cache = max_cache
        self._cache = OrderedDict()
        
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        self.vol_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        # Load metadata
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            volume_info = meta.get('volume_info', {})
            self.num_classes = meta.get('num_classes', 14)
            self.class_names = meta.get('class_names', [])
        else:
            volume_info = None
            self.num_classes = 14
            self.class_names = ['Background', 'Spleen', 'R.Kidney', 'L.Kidney', 'Gallbladder',
                               'Esophagus', 'Liver', 'Stomach', 'Aorta', 'IVC',
                               'Portal Vein', 'Pancreas', 'R.AG', 'L.AG']
        
        # Build index map
        self.index_map = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                vol = np.load(vp, mmap_mode='r')
                n_slices = vol.shape[2]
            for s in range(n_slices):
                self.index_map.append((i, s))
        
        print(f"SynapseDataset2D: {len(self.index_map)} slices from {len(self.vol_paths)} volumes")
    
    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        
        mode = 'r' if self.use_memmap else None
        vol = np.load(self.vol_paths[idx], mmap_mode=mode)
        mask = np.load(self.mask_paths[idx], mmap_mode=mode)
        self._cache[idx] = (vol, mask)
        
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return vol, mask
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        
        # vol shape: (H, W, D)
        # mask shape: (H, W, D)
        img = vol[:, :, slice_idx].copy().astype(np.float32)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        
        if self.in_channels == 3:
            img = img.repeat(3, 1, 1)  # (3, H, W)
        
        return img, torch.from_numpy(gt)
