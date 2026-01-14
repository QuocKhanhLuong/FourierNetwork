"""
Preprocess Synapse dataset: Multi-organ CT → .npy volumes
Classes: 14 organs (Spleen, Kidneys, Gallbladder, Liver, Stomach, Aorta, etc.)

Usage:
    python scripts/preprocess_synapse.py --input data/Synapse/training --output preprocessed_data/Synapse/training
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_ct(image, window_center=40, window_width=400):
    """CT windowing normalization for abdominal."""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)


def preprocess_patient(patient_path, target_size=(224, 224)):
    """Process one Synapse case."""
    patient_folder = os.path.basename(patient_path)
    
    img_path, label_path = None, None
    
    for f in os.listdir(patient_path):
        full_path = os.path.join(patient_path, f)
        if f.endswith(('.nii.gz', '.nii')):
            f_lower = f.lower()
            if 'label' in f_lower or 'seg' in f_lower or 'gt' in f_lower:
                label_path = full_path
            elif 'img' in f_lower or label_path is None:
                img_path = full_path
    
    if not img_path or not label_path:
        return []
    
    try:
        img_data = normalize_ct(nib.load(img_path).get_fdata())
        mask_data = nib.load(label_path).get_fdata().astype(np.uint8)
        
        num_slices = img_data.shape[2]
        
        resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            resized_img[:, :, i] = resize(img_data[:, :, i], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
            resized_mask[:, :, i] = resize(mask_data[:, :, i].astype(np.float32), target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
        
        return [(resized_img, resized_mask, patient_folder)]
    except Exception as e:
        print(f"  Error: {patient_folder}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Preprocess Synapse dataset')
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
    
    print(f"Synapse Preprocessing: {len(patient_folders)} cases")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for patient_path in tqdm(patient_folders, desc="Synapse"):
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
        'dataset': 'Synapse', 'num_classes': 14,
        'class_names': ['Background', 'Spleen', 'R.Kidney', 'L.Kidney', 'Gallbladder',
                        'Esophagus', 'Liver', 'Stomach', 'Aorta', 'IVC',
                        'Portal Vein', 'Pancreas', 'R.AG', 'L.AG'],
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
