"""
ACDC Training Script with Config File Support
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.egm_net import EGMNet
from data.acdc_dataset import ACDCDataset2D


CLASS_MAP = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device, num_classes=4):
    model.eval()
    tp, fp, fn = [0]*num_classes, [0]*num_classes, [0]*num_classes
    dice_s, iou_s = [0.]*num_classes, [0.]*num_classes
    batches, total_correct, total_pixels = 0, 0, 0
    
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
    parser.add_argument('--config', type=str, default='config.yaml')
    
    # Override config via CLI
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='weights')
    
    # Module toggles
    parser.add_argument('--use_dog', action='store_true', default=None)
    parser.add_argument('--no_dog', dest='use_dog', action='store_false')
    parser.add_argument('--use_mamba', action='store_true', default=None)
    parser.add_argument('--no_mamba', dest='use_mamba', action='store_false')
    parser.add_argument('--use_spectral', action='store_true', default=None)
    parser.add_argument('--no_spectral', dest='use_spectral', action='store_false')
    parser.add_argument('--use_fine_head', action='store_true', default=None)
    parser.add_argument('--no_fine_head', dest='use_fine_head', action='store_false')
    parser.add_argument('--fine_head_type', type=str, default=None, choices=['gabor', 'shearlet', 'none'])
    parser.add_argument('--coarse_head_type', type=str, default=None, choices=['constellation', 'linear', 'conv'])
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Merge config with CLI args (CLI takes priority)
    model_cfg = cfg.get('model', {})
    heads_cfg = cfg.get('heads', {})
    train_cfg = cfg.get('training', {})
    retinal_cfg = cfg.get('retinal', {})
    data_cfg = cfg.get('data', {})
    
    # Model params
    in_channels = model_cfg.get('in_channels', 3)
    num_classes = model_cfg.get('num_classes', 4)
    img_size = model_cfg.get('img_size', 224)
    base_channels = model_cfg.get('backbone', {}).get('base_channels', 32)
    
    # Component toggles
    use_mamba = args.use_mamba if args.use_mamba is not None else model_cfg.get('components', {}).get('use_mamba', False)
    use_spectral = args.use_spectral if args.use_spectral is not None else model_cfg.get('components', {}).get('use_spectral', False)
    use_dog = args.use_dog if args.use_dog is not None else retinal_cfg.get('enabled', False)
    
    # Head configs
    fine_enabled = args.use_fine_head if args.use_fine_head is not None else heads_cfg.get('fine', {}).get('enabled', True)
    fine_type = args.fine_head_type if args.fine_head_type else heads_cfg.get('fine', {}).get('type', 'gabor')
    coarse_type = args.coarse_head_type if args.coarse_head_type else heads_cfg.get('coarse', {}).get('type', 'constellation')
    
    # Training params
    epochs = args.epochs if args.epochs else train_cfg.get('epochs', 100)
    batch_size = args.batch_size if args.batch_size else train_cfg.get('batch_size', 8)
    lr = args.lr if args.lr else train_cfg.get('learning_rate', 1e-4)
    early_stop = train_cfg.get('early_stop', 20)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    data_dir = args.data_dir if args.data_dir else data_cfg.get('root', 'preprocessed_data/ACDC')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print config
    print(f"\n{'='*70}")
    print("ACDC Training with Config")
    print(f"{'='*70}")
    print(f"\n--- Model Config ---")
    print(f"  In Channels:    {in_channels}")
    print(f"  Num Classes:    {num_classes}")
    print(f"  Image Size:     {img_size}")
    print(f"\n--- Components ---")
    print(f"  DoG Retinal:    {'✓' if use_dog else '✗'}")
    print(f"  Mamba SSM:      {'✓' if use_mamba else '✗'}")
    print(f"  Spectral:       {'✓' if use_spectral else '✗'}")
    print(f"\n--- Heads ---")
    print(f"  Coarse Head:    {coarse_type}")
    print(f"  Fine Head:      {fine_type if fine_enabled else 'disabled'}")
    print(f"\n--- Training ---")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Learning Rate:  {lr}")
    print(f"  Data Dir:       {data_dir}")
    
    # Build model
    model = EGMNet(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        base_channels=base_channels,
        use_hrnet=True,
        use_mamba=use_mamba,
        use_spectral=use_spectral,
        use_fine_head=fine_enabled,
        use_dog=use_dog,
        fine_head_type=fine_type if fine_enabled else 'gabor',
        coarse_head_type=coarse_type
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters:     {params:,}")
    
    # Data
    train_dir = os.path.join(data_dir, 'training')
    test_dir = os.path.join(data_dir, 'testing')
    
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
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\n--- Data ---")
    print(f"  Train: {len(train_ds)} slices ({len(train_vols)} volumes)")
    print(f"  Val:   {len(val_ds)} slices ({len(val_vols)} volumes)")
    print(f"  Test:  {len(test_dataset)} slices")
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, 10)
    criterion = nn.CrossEntropyLoss()
    
    best_dice = 0
    epochs_no_improve = 0
    
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        valid_batches = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            out = model(imgs)['output']
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                continue
            
            loss = criterion(out, masks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1
        
        train_loss /= max(valid_batches, 1)
        
        torch.cuda.empty_cache()
        v = evaluate(model, val_loader, device, num_classes)
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{epochs} | Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*70}")
        
        print(f"{'Class':<6} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-"*50)
        for c in range(num_classes):
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
        
        if epochs_no_improve >= early_stop:
            print(f"\nEarly stop at epoch {epoch+1}")
            break
    
    # Test
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(f"{args.save_dir}/best.pt"))
    t = evaluate(model, test_loader, device, num_classes)
    
    print(f"{'Class':<6} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*50)
    for c in range(num_classes):
        print(f"{CLASS_MAP[c]:<6} {t['dice'][c]:>8.4f} {t['iou'][c]:>8.4f} {t['prec'][c]:>8.4f} {t['rec'][c]:>8.4f} {t['f1'][c]:>8.4f}")
    print("-"*50)
    print(f"{'AvgFG':<6} {t['mean_dice']:>8.4f} {t['mean_iou']:>8.4f} {t['mean_prec']:>8.4f} {t['mean_rec']:>8.4f} {t['mean_f1']:>8.4f}")
    
    print(f"\n✓ Done! Best Val Dice: {best_dice:.4f}, Test Dice: {t['mean_dice']:.4f}")


if __name__ == '__main__':
    main()
