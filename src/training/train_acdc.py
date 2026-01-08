"""
ACDC Training Script - Complete with Full Metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D


CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def evaluate(model, loader, device, num_classes=4):
    model.eval()
    tp = [0]*num_classes
    fp = [0]*num_classes
    fn = [0]*num_classes
    dice_s = [0.]*num_classes
    iou_s = [0.]*num_classes
    batches = 0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc="Eval", leave=False):
            imgs, tgts = imgs.to(device), tgts.to(device)
            out = model(imgs)['output']
            preds = out.argmax(1)
            batches += 1
            total_correct += (preds == tgts).sum().item()
            total_pixels += tgts.numel()
            
            for c in range(num_classes):
                pc = (preds == c).float().view(-1)
                tc = (tgts == c).float().view(-1)
                inter = (pc * tc).sum()
                dice_s[c] += ((2.*inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item()
                iou_s[c] += ((inter + 1e-6) / (pc.sum() + tc.sum() - inter + 1e-6)).item()
                tp[c] += inter.item()
                fp[c] += (pc.sum() - inter).item()
                fn[c] += (tc.sum() - inter).item()
    
    m = {'acc': total_correct/max(total_pixels, 1), 'dice': [], 'iou': [], 'prec': [], 'rec': [], 'f1': []}
    for c in range(num_classes):
        m['dice'].append(dice_s[c] / max(batches, 1))
        m['iou'].append(iou_s[c] / max(batches, 1))
        p = tp[c] / (tp[c] + fp[c] + 1e-6)
        r = tp[c] / (tp[c] + fn[c] + 1e-6)
        m['prec'].append(p)
        m['rec'].append(r)
        m['f1'].append(2*p*r / (p + r + 1e-6) if p + r > 0 else 0)
    
    m['mean_dice'] = np.mean(m['dice'][1:])
    m['mean_iou'] = np.mean(m['iou'][1:])
    m['mean_prec'] = np.mean(m['prec'][1:])
    m['mean_rec'] = np.mean(m['rec'][1:])
    m['mean_f1'] = np.mean(m['f1'][1:])
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--use_dog', action='store_true')
    parser.add_argument('--fine_head_type', type=str, default='gabor', choices=['gabor', 'shearlet'])
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='weights')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Model
    print(f"\n{'='*70}")
    print("ACDC Training - DoG + HRNet + Shearlet")
    print(f"{'='*70}")
    
    model = EGMNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        img_size=args.img_size,
        use_hrnet=True,
        use_mamba=False,
        use_spectral=False,
        use_fine_head=True,
        use_dog=args.use_dog,
        fine_head_type=args.fine_head_type
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")
    print(f"DoG: {'✓' if args.use_dog else '✗'}")
    print(f"Fine Head: {args.fine_head_type}")
    
    # Data
    train_dir = os.path.join(args.data_dir, 'training')
    test_dir = os.path.join(args.data_dir, 'testing')
    
    train_dataset = ACDCDataset2D(train_dir, in_channels=args.in_channels)
    test_dataset = ACDCDataset2D(test_dir, in_channels=args.in_channels)
    
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
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nTrain: {len(train_ds)} slices from {len(train_vols)} volumes")
    print(f"Val: {len(val_ds)} slices from {len(val_vols)} volumes")
    print(f"Test: {len(test_dataset)} slices")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, 10)
    criterion = nn.CrossEntropyLoss()
    
    best_dice = 0
    epochs_no_improve = 0
    
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"E{epoch+1} Train", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            out = model(imgs)['output']
            
            # Check for NaN
            if torch.isnan(out).any():
                print(f"Warning: NaN in output, skipping batch")
                continue
            
            loss = criterion(out, masks)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= max(len(train_loader), 1)
        
        # Eval
        torch.cuda.empty_cache()
        v = evaluate(model, val_loader, device, args.num_classes)
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*70}")
        
        print(f"{'Class':<6} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-"*50)
        for c in range(args.num_classes):
            print(f"{CLASS_MAP[c]:<6} {v['dice'][c]:>8.4f} {v['iou'][c]:>8.4f} {v['prec'][c]:>8.4f} {v['rec'][c]:>8.4f} {v['f1'][c]:>8.4f}")
        print("-"*50)
        print(f"{'AvgFG':<6} {v['mean_dice']:>8.4f} {v['mean_iou']:>8.4f} {v['mean_prec']:>8.4f} {v['mean_rec']:>8.4f} {v['mean_f1']:>8.4f}")
        print(f"Accuracy: {v['acc']:.4f}")
        
        scheduler.step(v['mean_dice'])
        
        if v['mean_dice'] > best_dice:
            best_dice = v['mean_dice']
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")
            print(f"★ Best saved! Dice={best_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        torch.save(model.state_dict(), f"{args.save_dir}/epoch_{epoch+1}.pt")
        
        if epochs_no_improve >= args.early_stop:
            print(f"\nEarly stop at epoch {epoch+1}")
            break
    
    # Test
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(f"{args.save_dir}/best.pt"))
    t = evaluate(model, test_loader, device, args.num_classes)
    
    print(f"{'Class':<6} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*50)
    for c in range(args.num_classes):
        print(f"{CLASS_MAP[c]:<6} {t['dice'][c]:>8.4f} {t['iou'][c]:>8.4f} {t['prec'][c]:>8.4f} {t['rec'][c]:>8.4f} {t['f1'][c]:>8.4f}")
    print("-"*50)
    print(f"{'AvgFG':<6} {t['mean_dice']:>8.4f} {t['mean_iou']:>8.4f} {t['mean_prec']:>8.4f} {t['mean_rec']:>8.4f} {t['mean_f1']:>8.4f}")
    
    print(f"\n✓ Done! Best Val Dice: {best_dice:.4f}, Test Dice: {t['mean_dice']:.4f}")


if __name__ == '__main__':
    main()
