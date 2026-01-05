"""
Physics-Inspired Dual Loss Function
Combines spatial (Dice + CE) and frequency domain losses for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice Loss for binary/multi-class segmentation."""
    
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing constant to avoid division by zero
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W) or (B, C, H, W)
            
        Returns:
            Scalar Dice loss
        """
        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)
        
        # Ensure target has same shape as pred for multi-class
        if target.ndim == 3:  # (B, H, W) -> convert to one-hot
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute Dice score
        intersection = torch.sum(pred * target, dim=2)
        union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        loss = 1.0 - dice
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class 1
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Scalar Focal loss
        """
        # Get class probabilities
        p = torch.softmax(pred, dim=1)
        
        # Get class log probabilities
        ce = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Get probability of true class
        p_t = torch.gather(p, 1, target.long().unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FrequencyLoss(nn.Module):
    """
    Frequency Domain Loss for enforcing edge sharpness and detail preservation.
    Computes L2 distance between FFT of prediction and ground truth.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize FrequencyLoss.
        
        Args:
            weight: Weight factor for this loss component in combined loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency domain loss.
        
        Uses FFT to compare frequency components, emphasizing edge preservation.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W) or (B, H, W)
            target: Ground truth tensor of same shape as pred
            
        Returns:
            Scalar frequency loss
        """
        # Ensure both have batch and channel dimensions
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        
        # Flatten to single channel for FFT comparison
        if pred.shape[1] > 1:
            # For multi-channel, convert to grayscale by averaging
            pred = pred.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)
        
        # Apply FFT to convert to frequency domain
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        
        # Compute L2 distance in frequency domain
        # Consider both magnitude and phase information
        loss_real = F.mse_loss(pred_freq.real, target_freq.real, reduction='mean')
        loss_imag = F.mse_loss(pred_freq.imag, target_freq.imag, reduction='mean')
        
        return loss_real + loss_imag


class SpectralDualLoss(nn.Module):
    """
    Combined Spectral Dual Loss.
    
    Combines:
    1. Spatial losses (Dice + Focal CE) - for overall shape and class balance
    2. Frequency loss - for edge sharpness and boundary preservation
    """
    
    def __init__(self, spatial_weight: float = 1.0, freq_weight: float = 0.1,
                 use_dice: bool = True, use_focal: bool = True):
        """
        Initialize SpectralDualLoss.
        
        Args:
            spatial_weight: Weight for spatial losses
            freq_weight: Weight for frequency loss
            use_dice: Whether to include Dice loss
            use_focal: Whether to include Focal loss (else CrossEntropy)
        """
        super().__init__()
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        self.use_dice = use_dice
        self.use_focal = use_focal
        
        # Spatial losses
        if use_dice:
            self.dice_loss = DiceLoss(smooth=1e-5)
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Frequency loss
        self.freq_loss = FrequencyLoss(weight=freq_weight)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            return_components: If True, return dict with individual loss components
            
        Returns:
            Scalar combined loss, or dict if return_components=True
        """
        # Ensure target is on same device as pred
        target = target.to(pred.device)
        
        # Spatial losses
        spatial_loss = 0.0
        losses_dict = {}
        
        if self.use_dice:
            dice = self.dice_loss(pred, target)
            spatial_loss = spatial_loss + dice
            losses_dict['dice'] = dice.item()
        
        if self.use_focal:
            focal = self.focal_loss(pred, target)
            spatial_loss = spatial_loss + focal
            losses_dict['focal'] = focal.item()
        else:
            ce = self.ce_loss(pred, target)
            spatial_loss = spatial_loss + ce
            losses_dict['ce'] = ce.item()
        
        # Frequency loss
        # For frequency loss, we need to extract the predicted class (argmax) and compare
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)  # (B, H, W)
        
        freq = self.freq_loss(pred_class.float(), target.float())
        losses_dict['freq'] = freq.item()
        
        # Weighted combination
        total_loss = (self.spatial_weight * spatial_loss + 
                     self.freq_weight * freq)
        losses_dict['total'] = total_loss.item()
        
        if return_components:
            return total_loss, losses_dict
        else:
            return total_loss


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes pixels near segmentation boundaries.
    Useful for improving edge precision.
    """
    
    def __init__(self, kernel_size: int = 3, weight: float = 1.0):
        """
        Initialize BoundaryAwareLoss.
        
        Args:
            kernel_size: Size of kernel for computing boundary gradients
            weight: Weight for boundary loss
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = weight
    
    def _compute_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary mask using gradient.
        
        Args:
            mask: Binary mask of shape (B, H, W)
            
        Returns:
            Boundary map of shape (B, H, W)
        """
        # Convert to float
        mask = mask.float().unsqueeze(1)  # (B, 1, H, W)
        
        # Compute gradients using Sobel-like operation
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=mask.dtype, device=mask.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=mask.dtype, device=mask.device)
        
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(mask, kernel_x, padding=1)
        grad_y = F.conv2d(mask, kernel_y, padding=1)
        
        # Compute magnitude of gradient
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Threshold to get boundary pixels
        boundary = (grad_magnitude > 0).float().squeeze(1)
        
        return boundary
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.
        
        Args:
            pred: Prediction logits of shape (B, C, H, W)
            target: Ground truth labels of shape (B, H, W)
            
        Returns:
            Scalar loss emphasizing boundaries
        """
        # Get predicted class
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)  # (B, H, W)
        
        # Compute boundary maps
        pred_boundary = self._compute_boundaries(pred_class)
        target_boundary = self._compute_boundaries(target)
        
        # Compute cross-entropy loss weighted by boundary
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Apply boundary weight (higher loss for boundary pixels)
        boundary_weight = (pred_boundary + target_boundary).clamp(0, 1)
        boundary_weight = 1.0 + boundary_weight  # Weight between 1 and 2
        
        weighted_loss = ce_loss * boundary_weight
        
        return weighted_loss.mean()


class EyeOpeningLoss(nn.Module):
    """
    Eye Opening Loss (Entropy Minimization at Boundaries).
    
    Based on the Eye Diagram concept from telecommunications:
    - Clear "eye" opening = decisive predictions (0 or 1)
    - Closed "eye" = uncertain predictions (~0.5)
    
    This loss penalizes predictions near 0.5 at boundary regions,
    forcing the model to make decisive (sharp) boundary predictions.
    
    L_eye = 4 × p × (1 - p)
    
    This reaches maximum (1.0) when p=0.5 and minimum (0.0) when p=0 or p=1.
    
    Features:
    - Warm-up scheduling: Only activates after N epochs
    - Annealing: Weight gradually increases to prevent early destabilization
    - Boundary focus: Optionally weight by energy/boundary map
    
    Args:
        warmup_epochs: Number of epochs before activating this loss
        max_weight: Maximum weight for this loss component
        anneal_rate: Rate of weight increase per epoch after warm-up
        smooth: Smoothing constant for numerical stability
    """
    
    def __init__(self, warmup_epochs: int = 5, max_weight: float = 0.1,
                 anneal_rate: float = 0.02, smooth: float = 1e-7):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.anneal_rate = anneal_rate
        self.smooth = smooth
    
    def get_weight(self, epoch: int) -> float:
        """
        Compute loss weight based on current epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Loss weight (0 during warm-up, then annealed up to max_weight)
        """
        if epoch < self.warmup_epochs:
            return 0.0
        return min(self.max_weight, self.anneal_rate * (epoch - self.warmup_epochs))
    
    def forward(self, logits: torch.Tensor, 
                energy_map: Optional[torch.Tensor] = None,
                epoch: int = 0) -> torch.Tensor:
        """
        Compute Eye Opening Loss.
        
        Args:
            logits: Prediction logits of shape (B, C, H, W) or (B, N, C)
            energy_map: Optional energy map (B, 1, H, W) to focus on boundaries
            epoch: Current training epoch for weight scheduling
            
        Returns:
            Scalar eye opening loss
        """
        # Get weight for current epoch
        weight = self.get_weight(epoch)
        
        if weight <= 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)
        
        # Convert logits to probabilities
        if logits.dim() == 4:
            # (B, C, H, W) -> softmax over classes
            probs = torch.softmax(logits, dim=1)
            # For multi-class, we want high confidence for ANY class
            # Maximum probability per pixel (how confident the prediction is)
            max_probs = probs.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        else:
            # (B, N, C) -> softmax over classes
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True)[0]  # (B, N, 1)
        
        # Eye opening: penalize uncertainty (probs near 0.5)
        # L = 4 × p × (1 - p), maximizes at p=0.5
        eye_loss = 4 * max_probs * (1 - max_probs)
        
        # Apply energy weighting (focus on boundaries) if provided
        if energy_map is not None:
            if energy_map.shape[-2:] != eye_loss.shape[-2:]:
                energy_map = F.interpolate(
                    energy_map, size=eye_loss.shape[-2:],
                    mode='bilinear', align_corners=True
                )
            eye_loss = eye_loss * energy_map
        
        # Return weighted mean
        return weight * eye_loss.mean()


class EGMCombinedLoss(nn.Module):
    """
    Combined Loss for EGM-Net Training.
    
    Combines:
    1. Coarse Loss: Dice + CE for the coarse segmentation
    2. Fine Loss: BCE/CE for point-sampled fine predictions
    3. Eye Opening Loss: Entropy minimization at boundaries (with warm-up)
    4. Consistency Loss: Agreement between coarse and fine branches
    
    Args:
        dice_weight: Weight for Dice loss
        ce_weight: Weight for CrossEntropy loss
        fine_weight: Weight for fine branch loss
        eye_weight: Max weight for Eye Opening loss
        consistency_weight: Weight for coarse-fine consistency
        eye_warmup: Epochs before activating Eye Opening loss
    """
    
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 fine_weight: float = 1.0, eye_weight: float = 0.1,
                 consistency_weight: float = 0.1, eye_warmup: int = 5):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.fine_weight = fine_weight
        self.eye_weight = eye_weight
        self.consistency_weight = consistency_weight
        
        # Loss components
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.eye_loss = EyeOpeningLoss(
            warmup_epochs=eye_warmup, 
            max_weight=eye_weight,
            anneal_rate=0.02
        )
    
    def forward(self, outputs: dict, target: torch.Tensor,
                point_logits: Optional[torch.Tensor] = None,
                point_labels: Optional[torch.Tensor] = None,
                epoch: int = 0) -> dict:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict with 'coarse', 'fine', 'energy'
            target: Ground truth mask (B, H, W)
            point_logits: Optional point predictions (B, N, C)
            point_labels: Optional point labels (B, N)
            epoch: Current training epoch
            
        Returns:
            Dict with 'total' loss and individual components
        """
        losses = {}
        
        # 1. Coarse branch losses
        coarse = outputs.get('coarse', outputs.get('output'))
        
        # Resize target if needed
        if coarse.shape[-2:] != target.shape[-2:]:
            target_resized = F.interpolate(
                target.unsqueeze(1).float(),
                size=coarse.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target.long()
        
        dice = self.dice_loss(coarse, target_resized)
        ce = self.ce_loss(coarse, target_resized)
        
        losses['dice'] = dice
        losses['ce'] = ce
        
        coarse_loss = self.dice_weight * dice + self.ce_weight * ce
        
        # 2. Fine branch loss (point-based if available)
        fine_loss = torch.tensor(0.0, device=coarse.device)
        if point_logits is not None and point_labels is not None:
            # Point-based cross-entropy
            B, N, C = point_logits.shape
            point_logits_flat = point_logits.view(B * N, C)
            point_labels_flat = point_labels.view(B * N)
            fine_loss = F.cross_entropy(point_logits_flat, point_labels_flat)
            losses['fine'] = fine_loss
        elif 'fine' in outputs:
            # Use spatial fine output
            fine = outputs['fine']
            if fine.shape[-2:] != target_resized.shape[-2:]:
                target_for_fine = F.interpolate(
                    target_resized.unsqueeze(1).float(),
                    size=fine.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_for_fine = target_resized
            fine_loss = self.ce_loss(fine, target_for_fine)
            losses['fine'] = fine_loss
        
        # 3. Eye Opening Loss (with warm-up)
        energy_map = outputs.get('energy', None)
        if 'fine' in outputs:
            eye = self.eye_loss(outputs['fine'], energy_map, epoch)
        else:
            eye = self.eye_loss(coarse, energy_map, epoch)
        losses['eye'] = eye
        
        # 4. Consistency Loss (coarse and fine should agree)
        consistency_loss = torch.tensor(0.0, device=coarse.device)
        if 'fine' in outputs and self.consistency_weight > 0:
            fine = outputs['fine']
            if fine.shape[-2:] != coarse.shape[-2:]:
                fine_resized = F.interpolate(
                    fine, size=coarse.shape[-2:],
                    mode='bilinear', align_corners=True
                )
            else:
                fine_resized = fine
            
            # KL divergence for consistency
            coarse_probs = F.log_softmax(coarse, dim=1)
            fine_probs = F.softmax(fine_resized, dim=1)
            consistency_loss = F.kl_div(coarse_probs, fine_probs, reduction='batchmean')
            losses['consistency'] = consistency_loss
        
        # Total loss
        total = (coarse_loss + 
                 self.fine_weight * fine_loss + 
                 eye + 
                 self.consistency_weight * consistency_loss)
        
        losses['total'] = total
        
        return losses



    # Test losses
    batch_size, num_classes, height, width = 2, 3, 64, 64
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test SpectralDualLoss
    loss_fn = SpectralDualLoss(spatial_weight=1.0, freq_weight=0.1)
    loss, components = loss_fn(pred, target, return_components=True)
    
    print(f"Total Loss: {loss.item():.4f}")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    
    # Test BoundaryAwareLoss
    boundary_loss_fn = BoundaryAwareLoss()
    boundary_loss = boundary_loss_fn(pred, target)
    print(f"\nBoundary Loss: {boundary_loss.item():.4f}")
