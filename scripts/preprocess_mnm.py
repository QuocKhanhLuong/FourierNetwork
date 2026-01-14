"""
Preprocess M&M dataset: Cardiac MRI multi-vendor → .npy volumes
Classes: 0=BG, 1=LV, 2=MYO, 3=RV

Usage:
    python scripts/preprocess_mnm.py --input data/MnM/Training --output preprocessed_data/MnM/training
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore(image):
    """Z-score normalization with outlier clipping."""
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / std if std > 0 else image - mean


def preprocess_patient(patient_path, target_size=(224, 224)):
    """Process one M&M patient."""
    patient_folder = os.path.basename(patient_path)
    
    img_path = os.path.join(patient_path, f'{patient_folder}_sa.nii.gz')
    mask_path = os.path.join(patient_path, f'{patient_folder}_sa_gt.nii.gz')
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return []
    
    try:
        img_data = nib.load(img_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()
        
        results = []
        
        # Handle 4D data
        if len(img_data.shape) == 4:
            for t in range(mask_data.shape[3]):
                if len(np.unique(mask_data[:, :, :, t])) > 1:
                    results.append((img_data[:, :, :, t], mask_data[:, :, :, t], f'{patient_folder}_t{t:02d}'))
        else:
            results.append((img_data, mask_data, patient_folder))
        
        processed = []
        for img_vol, mask_vol, vol_id in results:
            img_vol = normalize_zscore(np.squeeze(img_vol))
            mask_vol = np.squeeze(mask_vol).astype(np.uint8)
            
            if len(img_vol.shape) != 3:
                continue
            
            num_slices = img_vol.shape[2]
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(img_vol[:, :, i], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
                resized_mask[:, :, i] = resize(mask_vol[:, :, i].astype(np.float32), target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
            
            processed.append((resized_img, resized_mask, vol_id))
        
        return processed
    except Exception as e:
        print(f"  Error: {patient_folder}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Preprocess M&M dataset')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-skip', action='store_true')
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_folders = sorted([
        os.path.join(args.input, d) for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
    ])
    
    print(f"M&M Preprocessing: {len(patient_folders)} patients")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for patient_path in tqdm(patient_folders, desc="M&M"):
        for volume, mask, volume_id in preprocess_patient(patient_path, target_size):
            vol_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
            if not args.no_skip and os.path.exists(vol_path):
                skipped += 1
                continue
            
            np.save(vol_path, volume)
            np.save(mask_path, mask)
            volume_info[volume_id] = {'num_slices': int(mask.shape[2])}
            processed += 1
    
    metadata = {
        'dataset': 'MnM', 'num_classes': 4,
        'class_names': ['Background', 'LV', 'MYO', 'RV'],
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
