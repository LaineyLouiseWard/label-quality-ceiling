"""Knowledge Distillation utilities for model distillation and class remapping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Class orders all derive from the canonical source: geoseg/taxonomy.py. The KD mapping is
# grounded in the teacher's empirical confusion (build_mapping_from_confusion below), NOT a
# name-based hand-map.
from geoseg.taxonomy import OEM_NATIVE_CLASSES, STUDENT_CLASSES

# Native OpenEarthMap channel order (teacher output channel i == OEM class i). The KD targets
# (built below) follow Table 1's distillation column — rangeland and bareland carry semi-natural
# mass; the pre-training column is handled separately by relabel_oem_taxonomy.
OEM_CLASSES = list(OEM_NATIVE_CLASSES)

# Student output channel order — must match biodiversity_dataset.CLASSES exactly.
NEW_CLASSES = list(STUDENT_CLASSES)

# Tuple form for equality comparison against biodiversity_dataset.CLASSES.
REMAP_OUTPUT_CLASSES = tuple(NEW_CLASSES)

def build_mapping_from_confusion(mode="B", conf_path="artifacts/teacher_oem_gt_confusion.npz"):
    """Build the 9x6 KD mapping matrix GROUNDED in the teacher's training-set confusion.

    Reads the soft (prob-weighted) confusion saved by
    scripts/analysis/teacher_oem_to_gt_confusion.py and row-normalises it to
    P(GT student | teacher OEM) -- a soft label-transition matrix (cf. Patrini 2017
    forward-correction). This is the campaign KD map.

    Only mode 'B' (full data-driven: every OEM row = the measured distribution) is supported.
    The legacy name-based map and the partial mode 'A' were removed 2026-06-19.
    """
    if mode != "B":
        raise ValueError(f"only mode 'B' (grounded) is supported, got {mode!r}")
    data = np.load(conf_path, allow_pickle=True)
    soft = data["soft"].astype(np.float64)                       # (9, 6)
    rownorm = soft / soft.sum(axis=1, keepdims=True).clip(min=1e-12)
    return torch.tensor(rownorm, dtype=torch.float32)


class KDHelper:
    """Knowledge Distillation helper for computing teacher probabilities and KD loss."""
    
    def __init__(self, mapping_matrix, temperature=1.0):
        """Initialize KD helper.

        Args:
            mapping_matrix: torch.Tensor of shape (9, 6), maps OEM classes to student classes.
                Required — pass the grounded map from build_mapping_from_confusion("B").
            temperature: float, temperature for KD loss computation.
        """
        if mapping_matrix is None:
            raise ValueError(
                "KDHelper requires an explicit mapping_matrix; pass "
                'build_mapping_from_confusion("B") (the grounded KD map).'
            )
        self.mapping_matrix = mapping_matrix
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