
import torch
import torch.nn as nn
from models.hrnet_dcn import HRNetDCN
import argparse
from tabulate import tabulate

def analyze_hrnet_dcn():
    parser = argparse.ArgumentParser(description="Analyze HRNetDCN Model")
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--no_full_res', action='store_true', help='Disable full resolution')
    parser.add_argument('--use_pointrend', action='store_true')
    parser.add_argument('--use_shearlet', action='store_true')
    parser.add_argument('--deep_supervision', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"ANALYZING HRNetDCN MODEL STRUCTURE")
    print(f"{'='*60}")
    
    # Init Model
    model = HRNetDCN(
        in_channels=3,
        num_classes=4,
        base_channels=args.base_channels,
        use_pointrend=args.use_pointrend,
        full_resolution_mode=not args.no_full_res,
        deep_supervision=args.deep_supervision,
        use_shearlet=args.use_shearlet
    )
    
    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[1] PARAMETER COUNT")
    print(f"   - Total Params:     {total_params:,}")
    print(f"   - Trainable Params: {trainable_params:,}")
    print(f"   - Model Size (MB):  {total_params * 4 / 1024 / 1024:.2f} MB")

    # 2. Components Analysis
    print(f"\n[2] ACTIVE COMPONENTS")
    components = [
        ["Backbone", "HRNetDCN (Asymmetric)", "True"],
        ["Stream 1 Resolution", "224x224 (Full Res)" if not args.no_full_res else "56x56 (1/4)", "True"],
        ["Boundary Head", "PointRend", str(args.use_pointrend)],
        ["Refinement Head", "Shearlet", str(args.use_shearlet)],
        ["Aux Losses", "Deep Supervision (4 scales)", str(args.deep_supervision)]
    ]
    print(tabulate(components, headers=["Component", "Description", "Enabled"], tablefmt="grid"))
    
    # 3. Stage & Channel Details
    print(f"\n[3] ARCHITECTURE DETAILS (Base Channels = {args.base_channels})")
    
    C = args.base_channels
    if not args.no_full_res:
        # Full Res Mode
        stages = [
            ["Stem", "3 -> 64", "224x224 (1x)", "HRNetStemFullRes"],
            ["Layer1", "64 -> 256", "224x224 (1x)", "Bottleneck"],
            ["Stage2", f"[{C}, {C*2}]", "[1x, 1/2]", "BasicBlock (DCN)"],
            ["Stage3", f"[{C}, {C*2}, {C*4}]", "[1x, 1/2, 1/4]", "BasicBlock (DCN)"],
            ["Stage4", f"[{C}, {C*2}, {C*4}, {C*8}]", "[1x, 1/2, 1/4, 1/8]", "BasicBlock (DCN)"],
        ]
    else:
        # Standard Mode
        stages = [
            ["Stem", "3 -> 64", "56x56 (1/4)", "HRNetStem (Stride=4)"],
            ["Layer1", "64 -> 256", "56x56 (1/4)", "Bottleneck"],
            ["Stage2", f"[{C}, {C*2}]", "[1/4, 1/8]", "BasicBlock (DCN)"],
            ["Stage3", f"[{C}, {C*2}, {C*4}]", "[1/4, 1/8, 1/16]", "BasicBlock (DCN)"],
            ["Stage4", f"[{C}, {C*2}, {C*4}, {C*8}]", "[1/4, 1/8, 1/16, 1/32]", "BasicBlock (DCN)"],
        ]
    
    print(tabulate(stages, headers=["Stage", "Channels", "Resolution (Relative)", "Block Type"], tablefmt="grid"))

    # 4. Detailed Layer Params
    print(f"\n[4] DETAILED PARAMETERS BY SUBMODULE")
    submodules = [
        ("Stem", model.stem),
        ("Layer1", model.layer1),
        ("Stage2", model.stage2),
        ("Stage3", model.stage3),
        ("Stage4", model.stage4),
        ("Seg Head", model.seg_head),
    ]
    
    if args.use_pointrend:
        submodules.append(("PointRend", model.pointrend))
    if args.use_shearlet:
        submodules.append(("ShearletHead", model.shearlet_head))
    if args.deep_supervision:
        submodules.append(("AuxHeads", model.aux_heads))

    sub_data = []
    for name, module in submodules:
        params = sum(p.numel() for p in module.parameters())
        sub_data.append([name, f"{params:,}", f"{params/total_params*100:.1f}%"])
    
    print(tabulate(sub_data, headers=["Module", "Params", "% of Total"], tablefmt="grid"))
    print("\n")

if __name__ == '__main__':
    try:
        analyze_hrnet_dcn()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
