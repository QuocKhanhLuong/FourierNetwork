"""
Unified Preprocessing Script for Cardiac Datasets (ACDC, M&M)
Converts NIfTI files → .npy 3D volumes for fast memmap loading.

Features:
- Z-score normalization (robust for MRI)
- 3D volume output (compatible with ACDCDataset2D)
- Skip empty slices option
- Metadata.json generation
- Skip existing files

Usage:
    # ACDC
    python scripts/preprocess.py --dataset acdc \
        --input data/ACDC/training --output preprocessed_data/ACDC/training
    
    # M&M
    python scripts/preprocess.py --dataset mnm \
        --input data/MnM/Training --output preprocessed_data/MnM/training
"""

import os
import sys
import argparse
import configparser
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore(image):
    """
    Z-score normalization for MRI (more robust than max normalization).
    1. Clip outliers (0.5% - 99.5%)
    2. Subtract mean, divide by std
    """
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    return image - mean


def normalize_minmax(image):
    """Simple min-max normalization to [0, 1]."""
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image - min_val


# =============================================================================
# ACDC PREPROCESSING
# =============================================================================

def preprocess_single_patient_acdc(patient_path, target_size=(224, 224), normalize='zscore'):
    """Process one ACDC patient: load ED and ES frames, resize, return volumes."""
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')
    
    if not os.path.exists(info_cfg_path):
        return []
    
    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            config_string = '[DEFAULT]\n' + f.read()
        parser.read_string(config_string)
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error reading Info.cfg for {patient_folder}: {e}")
        return []
    
    results = []
    
    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'
        
        # Try both .nii.gz and .nii
        img_path = None
        mask_path = None
        
        for suffix in ['.gz', '']:
            test_img = os.path.join(patient_path, img_filename.replace('.gz', '') if suffix == '' else img_filename)
            test_mask = os.path.join(patient_path, mask_filename.replace('.gz', '') if suffix == '' else mask_filename)
            
            if os.path.exists(test_img):
                img_path = test_img
                mask_path = test_mask
                break
        
        if img_path is None or not os.path.exists(img_path):
            continue
        if not os.path.exists(mask_path):
            continue
        
        try:
            img_data = nib.load(img_path).get_fdata()
            mask_data = nib.load(mask_path).get_fdata()
            
            num_slices = img_data.shape[2]
            
            # Normalize entire volume first
            if normalize == 'zscore':
                img_data = normalize_zscore(img_data)
            else:
                img_data = normalize_minmax(img_data)
            
            # Resize slices
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(
                    img_data[:, :, i], target_size, 
                    order=1, preserve_range=True, anti_aliasing=True, mode='reflect'
                )
                resized_mask[:, :, i] = resize(
                    mask_data[:, :, i], target_size, 
                    order=0, preserve_range=True, anti_aliasing=False, mode='reflect'
                ).astype(np.uint8)
            
            volume_id = f"{patient_folder}_{frame_name}"
            results.append((resized_img, resized_mask, volume_id))
            
        except Exception as e:
            print(f"  Error processing {patient_folder} frame {frame_num}: {e}")
            continue
    
    return results


# =============================================================================
# M&M PREPROCESSING  
# =============================================================================

def preprocess_single_patient_mnm(patient_path, target_size=(224, 224), normalize='zscore'):
    """Process one M&M patient."""
    patient_folder = os.path.basename(patient_path)
    
    img_filename = f'{patient_folder}_sa.nii.gz'
    mask_filename = f'{patient_folder}_sa_gt.nii.gz'
    
    img_path = os.path.join(patient_path, img_filename)
    mask_path = os.path.join(patient_path, mask_filename)
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return []
    
    try:
        img_data = nib.load(img_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()
        
        results = []
        
        # Handle 4D data (multiple time frames)
        if len(img_data.shape) == 4 and len(mask_data.shape) == 4:
            for t in range(mask_data.shape[3]):
                mask_t = mask_data[:, :, :, t]
                if len(np.unique(mask_t)) > 1:  # Has labels
                    img_t = img_data[:, :, :, t]
                    results.append((img_t, mask_t, f'{patient_folder}_t{t:02d}'))
        elif len(img_data.shape) == 3 and len(mask_data.shape) == 3:
            results.append((img_data, mask_data, patient_folder))
        else:
            return []
        
        processed_results = []
        for img_vol, mask_vol, vol_id in results:
            img_vol = np.squeeze(img_vol)
            mask_vol = np.squeeze(mask_vol).astype(np.uint8)
            
            if len(img_vol.shape) != 3 or len(mask_vol.shape) != 3:
                continue
            
            # Match slices
            if img_vol.shape[2] != mask_vol.shape[2]:
                min_slices = min(img_vol.shape[2], mask_vol.shape[2])
                img_vol = img_vol[:, :, :min_slices]
                mask_vol = mask_vol[:, :, :min_slices]
            
            num_slices = img_vol.shape[2]
            
            # Normalize
            if normalize == 'zscore':
                img_vol = normalize_zscore(img_vol)
            else:
                img_vol = normalize_minmax(img_vol)
            
            # Resize
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(
                    img_vol[:, :, i], target_size,
                    order=1, preserve_range=True, anti_aliasing=True, mode='reflect'
                )
                resized_mask[:, :, i] = resize(
                    mask_vol[:, :, i].astype(np.float32), target_size,
                    order=0, preserve_range=True, anti_aliasing=False, mode='reflect'
                ).astype(np.uint8)
            
            processed_results.append((resized_img, resized_mask, vol_id))
        
        return processed_results
        
    except Exception as e:
        print(f"  Error processing {patient_folder}: {e}")
        return []


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_dataset(dataset, input_dir, output_dir, target_size=(224, 224), 
                       normalize='zscore', skip_existing=True):
    """
    Preprocess entire dataset.
    
    Args:
        dataset: 'acdc' or 'mnm'
        input_dir: Path to raw data
        output_dir: Path to save preprocessed .npy files
        target_size: Resize target
        normalize: 'zscore' or 'minmax'
        skip_existing: Skip if .npy already exists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get patient folders
    if dataset == 'acdc':
        patient_folders = sorted([
            os.path.join(input_dir, d) 
            for d in os.listdir(input_dir) 
            if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('patient')
        ])
        preprocess_func = preprocess_single_patient_acdc
        class_names = ['Background', 'RV', 'MYO', 'LV']
    else:  # mnm
        patient_folders = sorted([
            os.path.join(input_dir, d) 
            for d in os.listdir(input_dir) 
            if os.path.isdir(os.path.join(input_dir, d))
        ])
        preprocess_func = preprocess_single_patient_mnm
        class_names = ['Background', 'LV', 'MYO', 'RV']  # M&M label order
    
    if not patient_folders:
        print(f"No patient directories found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Preprocessing {dataset.upper()}")
    print(f"{'='*60}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Patients:   {len(patient_folders)}")
    print(f"Size:       {target_size}")
    print(f"Normalize:  {normalize}")
    print(f"{'='*60}\n")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for patient_path in tqdm(patient_folders, desc=f"Preprocessing {dataset.upper()}"):
        results = preprocess_func(patient_path, target_size, normalize)
        
        if not results:
            failed_count += 1
            continue
        
        for volume, mask, volume_id in results:
            volume_save_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_save_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
            # Skip existing
            if skip_existing and os.path.exists(volume_save_path) and os.path.exists(mask_save_path):
                try:
                    test_mask = np.load(mask_save_path)
                    volume_info[volume_id] = {
                        'num_slices': int(test_mask.shape[2]),
                        'target_size': list(target_size)
                    }
                    skipped_count += 1
                    continue
                except:
                    pass
            
            # Save
            try:
                np.save(volume_save_path, volume)
                np.save(mask_save_path, mask)
                
                volume_info[volume_id] = {
                    'num_slices': int(mask.shape[2]),
                    'target_size': list(target_size)
                }
                processed_count += 1
                
            except Exception as e:
                print(f"  Error saving {volume_id}: {e}")
                failed_count += 1
    
    # Save metadata
    metadata = {
        'dataset': dataset.upper(),
        'target_size': list(target_size),
        'normalize': normalize,
        'total_volumes': len(volume_info),
        'newly_processed': processed_count,
        'skipped_existing': skipped_count,
        'failed': failed_count,
        'volume_info': volume_info,
        'num_classes': 4,
        'class_names': class_names
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"{dataset.upper()} Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count} volumes")
    print(f"  Skipped:   {skipped_count} volumes (already exists)")
    print(f"  Failed:    {failed_count} patients")
    print(f"  Total:     {len(volume_info)} volumes")
    print(f"  Metadata:  {metadata_path}")
    print(f"{'='*60}")
    
    return processed_count, len(volume_info)


def main():
    parser = argparse.ArgumentParser(description='Preprocess cardiac datasets to .npy files')
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnm'],
                       help='Dataset type: acdc or mnm')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (e.g., data/ACDC/training)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory (e.g., preprocessed_data/ACDC/training)')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--normalize', type=str, default='zscore', choices=['zscore', 'minmax'],
                       help='Normalization method (default: zscore)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess even if .npy files exist')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        dataset=args.dataset,
        input_dir=args.input,
        output_dir=args.output,
        target_size=(args.size, args.size),
        normalize=args.normalize,
        skip_existing=not args.no_skip
    )


if __name__ == '__main__':
    main()
