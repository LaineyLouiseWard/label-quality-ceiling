"""Knowledge Distillation utilities for model distillation and class remapping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Original OEM classes (9)
OEM_CLASSES = [
    'Tree',          # 0 -> Forest
    'Rangeland',     # 1 -> Grassland (0.7) + SemiNatural (0.3)
    'Cropland',      # 2 -> Cropland
    'Developed',     # 3 -> Settlement
    'Road',          # 4 -> Settlement
    'Bareland',      # 5 -> SemiNatural
    'Water',         # 6 -> Background
    'Background',    # 7 -> Background
    'Unknown'        # 8 -> Background (unknown class)
]

# Output channel order — must match biodiversity_dataset.CLASSES exactly.
# Verified at config-parse time by the assertion in config/biodiversity/stage6_kd.py.
NEW_CLASSES = [
    'Background',   # 0
    'Forest',       # 1
    'Grassland',    # 2
    'Cropland',     # 3
    'Settlement',   # 4
    'Seminatural',  # 5
]

# Tuple form for equality comparison against biodiversity_dataset.CLASSES.
REMAP_OUTPUT_CLASSES = tuple(NEW_CLASSES)

def create_mapping_matrix(alpha=0.7, class_weights=None):
    """Create 9x6 mapping matrix M that maps OEM classes to new taxonomy.
    
    Args:
        alpha: float, portion of Rangeland that goes to Grassland (1-alpha goes to SemiNatural)
        class_weights: Optional list/tensor of weights for target classes to boost minority classes
        
    Returns:
        torch.Tensor of shape (9, 6) with each row summing to 1.0
    """
    # Initialize mapping matrix
    M = torch.zeros(len(OEM_CLASSES), len(NEW_CLASSES))
    
    # Column indices match NEW_CLASSES order:
    # Background=0, Forest=1, Grassland=2, Cropland=3, Settlement=4, Seminatural=5
    M[0, 1] = 1.0            # Tree        -> Forest      (col 1)
    M[1, 2] = alpha          # Rangeland   -> Grassland   (col 2, alpha)
    M[1, 5] = 1.0 - alpha   # Rangeland   -> Seminatural (col 5, 1-alpha)
    M[2, 3] = 1.0            # Cropland    -> Cropland    (col 3)
    M[3, 4] = 1.0            # Developed   -> Settlement  (col 4)
    M[4, 4] = 1.0            # Road        -> Settlement  (col 4)
    M[5, 5] = 1.0            # Bareland    -> Seminatural (col 5)
    M[6, 0] = 1.0            # Water       -> Background  (col 0)
    M[7, 0] = 1.0            # Background  -> Background  (col 0)
    M[8, 0] = 1.0            # Unknown     -> Background  (col 0)
    
    # Apply class weights to boost minority classes
    if class_weights is not None:
        if isinstance(class_weights, (list, tuple)):
            class_weights = torch.FloatTensor(class_weights)
        # Multiply each column by its weight
        M = M * class_weights.unsqueeze(0)
        # Re-normalize rows to sum to 1
        row_sums = M.sum(dim=1, keepdim=True)
        M = M / (row_sums + 1e-8)
    
    return M


class KDHelper:
    """Knowledge Distillation helper for computing teacher probabilities and KD loss."""
    
    def __init__(self, mapping_matrix=None, alpha=0.7, temperature=1.0):
        """Initialize KD helper.
        
        Args:
            mapping_matrix: torch.Tensor of shape (8, 6), maps old classes to new.
                If None, will create using create_mapping_matrix.
            alpha: float, portion of Rangeland that goes to Grassland if creating mapping.
            temperature: float, temperature for KD loss computation.
        """
        self.mapping_matrix = mapping_matrix if mapping_matrix is not None else create_mapping_matrix(alpha)
        self.temperature = temperature
        self.cache = {}  # Optional cache for teacher predictions
    
    def remap_teacher_probs(self, teacher_logits):
        """Convert 9-class teacher probabilities to 6-class using mapping matrix.
        
        Args:
            teacher_logits: torch.Tensor of shape (N, 9, H, W)
            
        Returns:
            torch.Tensor of shape (N, 6, H, W)
        """
        # Convert logits to probabilities
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Reshape to (N, 9, H*W)
        N, C, H, W = teacher_probs.shape
        teacher_probs = teacher_probs.view(N, C, -1)
        
        # Move mapping matrix to same device as teacher_probs
        mapping_matrix = self.mapping_matrix.to(teacher_probs.device)
        
        # Apply mapping using einsum: (N, 9, H*W) x (9, 6) -> (N, 6, H*W)
        # For each pixel, we multiply the 9-dim prob vector by the mapping matrix
        new_probs = torch.einsum('ncp,cd->ndp', teacher_probs, mapping_matrix)
        
        # Reshape back to (N, 6, H, W)
        return new_probs.view(N, len(NEW_CLASSES), H, W)
    
    def compute_kd_loss(self, student_logits, teacher_logits, class_weights=None, reduction='mean', confidence_threshold=None):
        """Compute KL divergence loss between student and remapped teacher.
        
        Args:
            student_logits: torch.Tensor (N, 6, H, W), student predictions
            teacher_logits: torch.Tensor (N, 9, H, W), teacher predictions
            class_weights: Optional tensor (6,) to weight different classes
            reduction: str, 'none', 'mean', or 'sum'
            confidence_threshold: Optional float, ignore teacher predictions where max prob < threshold (e.g., 0.5)
            
        Returns:
            torch.Tensor, KL divergence loss
        """
        # Get remapped teacher probabilities
        teacher_probs = self.remap_teacher_probs(teacher_logits)
        
        # Get student probabilities
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        
        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_probs,
            reduction='none'
        )
        
        # Apply confidence masking if threshold provided
        if confidence_threshold is not None:
            # Get teacher's max probability per pixel
            teacher_max_prob = teacher_probs.max(dim=1, keepdim=True)[0]  # (N, 1, H, W)
            
            # Create mask: 1 where teacher is confident, 0 where not
            confidence_mask = (teacher_max_prob >= confidence_threshold).float()
            
            # Apply mask to KD loss (ignore uncertain teacher predictions)
            kd_loss = kd_loss * confidence_mask
        
        # Apply class weights if provided
        if class_weights is not None:
            if class_weights.device != kd_loss.device:
                class_weights = class_weights.to(kd_loss.device)
            # Expand weights to match kd_loss shape: (1, C, 1, 1)
            weights = class_weights.view(1, -1, 1, 1)
            kd_loss = kd_loss * weights
        
        # Apply reduction
        if reduction == 'none':
            return kd_loss
        elif reduction == 'mean':
            return kd_loss.mean()
        else:  # sum
            return kd_loss.sum()
    
    def cache_teacher_probs(self, image_id, teacher_logits):
        """Cache teacher probabilities for an image."""
        self.cache[image_id] = self.remap_teacher_probs(teacher_logits).cpu()
    
    def get_cached_probs(self, image_id):
        """Retrieve cached teacher probabilities."""
        return self.cache.get(image_id)
    
    def clear_cache(self):
        """Clear the probability cache."""
        self.cache.clear()