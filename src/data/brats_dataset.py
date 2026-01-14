"""
BraTS21 Dataset Loader for 2D Training
Loads preprocessed .npy volumes (4 modalities: T1, T1ce, T2, FLAIR)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import json


class BraTSDataset2D(Dataset):
    """
    BraTS21 Dataset for 2D slice training.
    
    Args:
        npy_dir: Path to preprocessed data (with volumes/ and masks/ subdirs)
        use_memmap: Use memory-mapped loading
        in_channels: 4 (use all modalities) or 1 (use single modality)
        modality_idx: If in_channels=1, which modality (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
        max_cache: Max volumes to cache
    """
    
    def __init__(self, npy_dir, use_memmap=True, in_channels=4, modality_idx=None, max_cache=10):
        self.use_memmap = use_memmap
        self.in_channels = in_channels
        self.modality_idx = modality_idx
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
            self.num_classes = meta.get('num_classes', 4)
            self.class_names = meta.get('class_names', ['BG', 'NCR', 'ED', 'ET'])
        else:
            volume_info = None
            self.num_classes = 4
            self.class_names = ['BG', 'NCR', 'ED', 'ET']
        
        # Build index map: (vol_idx, slice_idx)
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
        
        print(f"BraTSDataset2D: {len(self.index_map)} slices from {len(self.vol_paths)} volumes")
    
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
        
        # vol shape: (H, W, D, 4)
        # mask shape: (H, W, D)
        img = vol[:, :, slice_idx, :].copy().astype(np.float32)  # (H, W, 4)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        
        # Transpose to (C, H, W)
        img = np.transpose(img, (2, 0, 1))  # (4, H, W)
        
        if self.in_channels == 1 and self.modality_idx is not None:
            img = img[self.modality_idx:self.modality_idx+1]
        elif self.in_channels == 3:
            # Use T1ce, T2, FLAIR (skip T1)
            img = img[1:4]
        
        return torch.from_numpy(img), torch.from_numpy(gt)
