"""
Preprocess BraTS21 dataset: Brain Tumor MRI → .npy volumes
4 modalities: T1, T1ce, T2, FLAIR
Classes: 0=BG, 1=NCR (necrotic), 2=ED (edema), 3=ET (enhancing tumor)
Note: Original label 4 is remapped to 3

Usage:
    python scripts/preprocess_brats.py --input data/BraTS21/training --output preprocessed_data/BraTS21/training
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore_masked(image, mask):
    """Z-score normalization within brain mask."""
    brain = image[mask > 0]
    if len(brain) > 0:
        mean, std = np.mean(brain), np.std(brain)
        if std > 0:
            return (image - mean) / std
        return image - mean
    return image


def preprocess_patient(patient_path, target_size=(224, 224)):
    """Process one BraTS21 patient with 4 modalities."""
    patient_folder = os.path.basename(patient_path)
    
    modalities = ['t1', 't1ce', 't2', 'flair']
    modality_files = {}
    seg_path = None
    
    for f in os.listdir(patient_path):
        f_lower = f.lower()
        if 'seg' in f_lower and f.endswith(('.nii.gz', '.nii')):
            seg_path = os.path.join(patient_path, f)
        else:
            for mod in modalities:
                if f_lower.endswith(f'{mod}.nii.gz') or f_lower.endswith(f'{mod}.nii'):
                    modality_files[mod] = os.path.join(patient_path, f)
    
    if not seg_path or len(modality_files) < 4:
        return []
    
    try:
        # Load all 4 modalities
        mod_data = [nib.load(modality_files[mod]).get_fdata() for mod in modalities]
        img_4d = np.stack(mod_data, axis=-1)  # (H, W, D, 4)
        
        mask_data = nib.load(seg_path).get_fdata().astype(np.uint8)
        mask_data[mask_data == 4] = 3  # Remap label 4 -> 3
        
        num_slices = img_4d.shape[2]
        
        # Normalize each modality within brain mask
        brain_mask = (img_4d[:, :, :, 0] > 0)
        for m in range(4):
            img_4d[:, :, :, m] = normalize_zscore_masked(img_4d[:, :, :, m], brain_mask)
        
        # Resize
        resized_img = np.zeros((target_size[0], target_size[1], num_slices, 4), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            for m in range(4):
                resized_img[:, :, i, m] = resize(img_4d[:, :, i, m], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
            resized_mask[:, :, i] = resize(mask_data[:, :, i], target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
        
        return [(resized_img, resized_mask, patient_folder)]
    except Exception as e:
        print(f"  Error: {patient_folder}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Preprocess BraTS21 dataset')
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
        if os.path.isdir(os.path.join(args.input, d)) and d.startswith('BraTS')
    ])
    
    print(f"BraTS21 Preprocessing: {len(patient_folders)} patients")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for patient_path in tqdm(patient_folders, desc="BraTS21"):
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
        'dataset': 'BraTS21', 'num_classes': 4, 'num_modalities': 4,
        'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
        'class_names': ['Background', 'NCR', 'ED', 'ET'],
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
