"""
ACDC Complete Training Script
Combines all features from previous training files:
- Boundary Loss with warmup
- Deep Supervision
- TTA for evaluation
- 3D Volumetric Metrics
- DCN Dilation Pyramid
- PointRend integration
- Point Sampling (optional)
- Mixed Precision (AMP)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime
from typing import Dict, Optional

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D
from losses.sota_loss import CombinedSOTALoss, TTAInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_2d(model, loader, device, num_classes=4, use_tta=False):
    """2D Slice-based evaluation with optional TTA."""
    model.eval()
    
    if use_tta:
        tta = TTAInference(model, device)
    
    dice_s = [0.] * num_classes
    iou_s = [0.] * num_classes
    batches = 0
    
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            
            if use_tta:
                preds = tta.predict_8x(imgs)
            else:
                out = model(imgs)['output']
                preds = out.argmax(1)
            
            batches += 1
            
            for c in range(num_classes):
                pc = (preds == c).float().view(-1)
                tc = (tgts == c).float().view(-1)
                inter = (pc * tc).sum()
                dice_s[c] += ((2. * inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc.sum() + tc.sum() - inter + 1e-6)).item()
    
    metrics = {'dice': [], 'iou': []}
    for c in range(num_classes):
        metrics['dice'].append(dice_s[c] / max(batches, 1))
        metrics['iou'].append(iou_s[c] / max(batches, 1))
    
    metrics['mean_dice'] = np.mean(metrics['dice'][1:])
    metrics['mean_iou'] = np.mean(metrics['iou'][1:])
    return metrics


def evaluate_3d(model, dataset, device, num_classes=4):
    """
    3D Volumetric evaluation - groups slices by volume.
    More accurate for medical imaging.
    """
    model.eval()
    
    vol_preds = defaultdict(list)
    vol_targets = defaultdict(list)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            vol_idx, slice_idx = dataset.dataset.index_map[dataset.indices[i]]
            
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)
            
            out = model(img)['output']
            pred = out.argmax(1).squeeze(0).cpu().numpy()
            target_np = target.numpy()
            
            vol_preds[vol_idx].append((slice_idx, pred))
            vol_targets[vol_idx].append((slice_idx, target_np))
    
    dice_3d = {c: [] for c in range(1, num_classes)}
    hd95_3d = {c: [] for c in range(1, num_classes)}
    
    for vol_idx in vol_preds.keys():
        preds_sorted = sorted(vol_preds[vol_idx], key=lambda x: x[0])
        targets_sorted = sorted(vol_targets[vol_idx], key=lambda x: x[0])
        
        pred_3d = np.stack([p[1] for p in preds_sorted], axis=0)
        target_3d = np.stack([t[1] for t in targets_sorted], axis=0)
        
        for c in range(1, num_classes):
            pred_c = (pred_3d == c)
            target_c = (target_3d == c)
            
            # 3D Dice
            inter = (pred_c & target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * inter) / (union + 1e-6)
            dice_3d[c].append(dice)
            
            # 3D HD95
            if pred_c.any() and target_c.any():
                pred_dist = distance_transform_edt(~pred_c)
                target_dist = distance_transform_edt(~target_c)
                
                pred_border = pred_c ^ binary_erosion(pred_c)
                target_border = target_c ^ binary_erosion(target_c)
                
                if pred_border.any() and target_border.any():
                    d1 = target_dist[pred_border]
                    d2 = pred_dist[target_border]
                    all_d = np.concatenate([d1, d2])
                    hd95_3d[c].append(np.percentile(all_d, 95))
                else:
                    hd95_3d[c].append(0.0)
            elif not pred_c.any() and not target_c.any():
                hd95_3d[c].append(0.0)
            else:
                hd95_3d[c].append(100.0)
    
    return {
        'mean_dice': np.mean([np.mean(dice_3d[c]) for c in range(1, num_classes)]),
        'mean_hd95': np.mean([np.mean(hd95_3d[c]) for c in range(1, num_classes)]),
        'per_class_dice': {c: np.mean(dice_3d[c]) for c in range(1, num_classes)},
        'per_class_hd95': {c: np.mean(hd95_3d[c]) for c in range(1, num_classes)},
        'num_volumes': len(vol_preds)
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=False):
    """Single training epoch with optional AMP."""
    model.train()
    
    train_loss = 0
    loss_dict_sum = {}
    valid_batches = 0
    
    pbar = tqdm(loader, desc=f"E{epoch+1}", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(imgs)['output']
                if torch.isnan(out).any():
                    continue
                loss, loss_dict = criterion(out, masks)
            
            if torch.isnan(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)['output']
            if torch.isnan(out).any():
                continue
            
            loss, loss_dict = criterion(out, masks)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        train_loss += loss.item()
        valid_batches += 1
        
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = train_loss / max(valid_batches, 1)
    avg_loss_dict = {k: v / max(valid_batches, 1) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ACDC Complete Training")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Model
    parser.add_argument('--block_type', type=str, default='dcn',
                       choices=['basic', 'convnext', 'dcn', 'inverted_residual', 'swin', 'fno', 'wavelet', 'rwkv'])
    parser.add_argument('--use_dog', action='store_true', help='Enable DoG preprocessing')
    parser.add_argument('--fine_head_type', type=str, default='shearlet', choices=['gabor', 'shearlet'])
    parser.add_argument('--use_mamba', action='store_true', help='Enable Mamba blocks')
    parser.add_argument('--use_spectral', action='store_true', help='Enable spectral processing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision training')
    
    # Loss
    parser.add_argument('--boundary_weight', type=float, default=0.5)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    
    # Evaluation
    parser.add_argument('--eval_3d', action='store_true', help='Use 3D volumetric evaluation')
    parser.add_argument('--use_tta', action='store_true', help='Enable TTA for validation')
    parser.add_argument('--tta_test', action='store_true', help='Enable 8x TTA for test')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for saving')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.block_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Config
    in_channels = 3
    num_classes = 4
    img_size = 224
    
    print(f"\n{'='*70}")
    print("ACDC Complete Training")
    print(f"{'='*70}")
    print(f"Experiment: {args.exp_name}")
    
    # Model
    model = EGMNet(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        use_hrnet=True,
        use_mamba=args.use_mamba,
        use_spectral=args.use_spectral,
        use_fine_head=True,
        use_dog=args.use_dog,
        fine_head_type=args.fine_head_type,
        block_type=args.block_type
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model ---")
    print(f"  Parameters:     {params:,}")
    print(f"  Block Type:     {args.block_type}")
    print(f"  DoG:            {'✓' if args.use_dog else '✗'}")
    print(f"  Mamba:          {'✓' if args.use_mamba else '✗'}")
    print(f"  Spectral:       {'✓' if args.use_spectral else '✗'}")
    print(f"  Fine Head:      {args.fine_head_type}")
    
    print(f"\n--- Training ---")
    print(f"  Epochs:         {args.epochs}")
    print(f"  LR:             {args.lr}")
    print(f"  Warmup:         {args.warmup_epochs} epochs")
    print(f"  Early Stop:     {args.early_stop} epochs")
    print(f"  AMP:            {'✓' if args.use_amp else '✗'}")
    
    print(f"\n--- Loss ---")
    print(f"  CE Weight:      {args.ce_weight}")
    print(f"  Dice Weight:    {args.dice_weight}")
    print(f"  Boundary:       {args.boundary_weight} (after warmup)")
    
    print(f"\n--- Evaluation ---")
    print(f"  Eval Mode:      {'3D Volumetric' if args.eval_3d else '2D Slice'}")
    print(f"  TTA Val:        {'✓' if args.use_tta else '✗'}")
    print(f"  TTA Test:       {'✓ 8x' if args.tta_test else '✗'}")
    
    # Data
    train_dir = os.path.join(args.data_dir, 'training')
    test_dir = os.path.join(args.data_dir, 'testing')
    
    train_dataset = ACDCDataset2D(train_dir, in_channels=in_channels)
    test_dataset = ACDCDataset2D(test_dir, in_channels=in_channels)
    
    # Volume-based split
    num_vols = len(train_dataset.vol_paths)
    vol_indices = list(range(num_vols))
    np.random.seed(42)
    np.random.shuffle(vol_indices)
    split = int(num_vols * 0.8)
    train_vols = set(vol_indices[:split])
    val_vols = set(vol_indices[split:])
    
    train_indices = [i for i, (v, s) in enumerate(train_dataset.index_map) if v in train_vols]
    val_indices = [i for i, (v, s) in enumerate(train_dataset.index_map) if v in val_vols]
    
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True)
    
    print(f"\n--- Data ---")
    print(f"  Train: {len(train_ds)} slices ({len(train_vols)} vols)")
    print(f"  Val:   {len(val_ds)} slices ({len(val_vols)} vols)")
    print(f"  Test:  {len(test_dataset)} slices")
    
    # Loss & Optimizer
    criterion = CombinedSOTALoss(
        num_classes=num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
        warmup_epochs=args.warmup_epochs
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    best_dice = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_dice': [], 'val_hd95': []}
    
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    for epoch in range(args.epochs):
        criterion.set_epoch(epoch)
        
        # Train
        train_loss, loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, scaler, args.use_amp
        )
        scheduler.step()
        
        # Evaluate
        torch.cuda.empty_cache()
        
        if args.eval_3d:
            v = evaluate_3d(model, val_ds, device, num_classes)
            val_dice = v['mean_dice']
            val_hd95 = v['mean_hd95']
        else:
            v = evaluate_2d(model, val_loader, device, num_classes, use_tta=args.use_tta)
            val_dice = v['mean_dice']
            val_hd95 = 0.0  # Not computed in 2D mode
        
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_hd95'].append(val_hd95)
        
        # Print
        lr = scheduler.get_last_lr()[0]
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        
        if args.eval_3d:
            print(f"E{epoch+1:03d} | LR: {lr:.6f} | {loss_str} | "
                  f"3D Dice: {val_dice:.4f} | HD95: {val_hd95:.2f}")
        else:
            print(f"E{epoch+1:03d} | LR: {lr:.6f} | {loss_str} | Dice: {val_dice:.4f}")
        
        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best.pt'))
            print(f"  ★ Best! Dice={best_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stop
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stop at epoch {epoch+1}")
            break
        
        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch_{epoch+1}.pt'))
    
    # Save final
    torch.save(model.state_dict(), os.path.join(exp_dir, 'last.pt'))
    
    # Save history
    np.save(os.path.join(exp_dir, 'history.npy'), history)
    
    # Test evaluation
    print(f"\n{'='*70}")
    print("TEST EVALUATION" + (" (8x TTA)" if args.tta_test else ""))
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best.pt'), weights_only=True))
    
    if args.eval_3d:
        # Create test subset wrapper
        test_indices = list(range(len(test_dataset)))
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        # Manually set attributes needed by evaluate_3d
        test_subset.dataset = test_dataset
        test_subset.indices = test_indices
        
        t = evaluate_3d(model, test_subset, device, num_classes)
        
        print(f"\n{'Class':<6} {'Dice':>8} {'HD95':>8}")
        print("-" * 30)
        for c in range(1, num_classes):
            print(f"{CLASS_MAP[c]:<6} {t['per_class_dice'][c]:>8.4f} {t['per_class_hd95'][c]:>8.2f}")
        print("-" * 30)
        print(f"{'AvgFG':<6} {t['mean_dice']:>8.4f} {t['mean_hd95']:>8.2f}")
    else:
        t = evaluate_2d(model, test_loader, device, num_classes, use_tta=args.tta_test)
        
        print(f"\n{'Class':<6} {'Dice':>8} {'IoU':>8}")
        print("-" * 30)
        for c in range(num_classes):
            print(f"{CLASS_MAP[c]:<6} {t['dice'][c]:>8.4f} {t['iou'][c]:>8.4f}")
        print("-" * 30)
        print(f"{'AvgFG':<6} {t['mean_dice']:>8.4f} {t['mean_iou']:>8.4f}")
    
    print(f"\n✓ Done!")
    print(f"  Best Val Dice: {best_dice:.4f}")
    print(f"  Test Dice:     {t['mean_dice']:.4f}")
    print(f"  Saved to:      {exp_dir}/")


if __name__ == '__main__':
    main()
