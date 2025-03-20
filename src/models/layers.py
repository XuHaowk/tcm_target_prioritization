"""
Custom layers for TCM target prioritization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class AttentionLayer(nn.Module):
    """Multi-head attention layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "Hidden dimension must be divisible by number of heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Node features.
            edge_index: Edge indices.
            edge_type: Edge types.
            
        Returns:
            Updated node features.
        """
        # Get dimensions
        batch_size, num_nodes, hidden_dim = node_features.size(0), node_features.size(0), self.hidden_dim
        
        # Project queries, keys, values
        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions [num_heads, num_nodes, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply edge mask
        if edge_index.size(1) > 0:
            # Create adjacency mask
            mask = torch.zeros(num_nodes, num_nodes, device=node_features.device)
            
            # Set mask to -inf for non-connected nodes
            mask.fill_(-1e9)
            
            # Set mask to 0 for connected nodes
            mask[edge_index[0], edge_index[1]] = 0
            
            # Apply mask to scores
            scores = scores + mask.unsqueeze(0)
        
        # Apply attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        context = torch.matmul(weights, v)
        
        # Transpose and reshape
        context = context.transpose(0, 1).contiguous().view(num_nodes, hidden_dim)
        
        # Apply output projection
        output = self.out_proj(context)
        
        return output

class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, hidden_dim: int, eps: float = 1e-12):
        """
        Initialize layer normalization.
        
        Args:
            hidden_dim: Hidden dimension.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Normalized tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma * x_normalized + self.beta

class GatedResidual(nn.Module):
    """Gated residual connection."""
    
    def __init__(self, hidden_dim: int):
        """
        Initialize gated residual connection.
        
        Args:
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            residual: Residual tensor.
            
        Returns:
            Gated output.
        """
        # Concatenate input and residual
        concat = torch.cat([x, residual], dim=-1)
        
        # Compute gate
        gate = torch.sigmoid(self.gate(concat))
        
        # Apply gate
        return gate * x + (1 - gate) * residual

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        """
        Initialize focal loss.
        
        Args:
            gamma: Focusing parameter.
            alpha: Class weight.
            reduction: Reduction method.
        """
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input logits.
            targets: Target labels.
            
        Returns:
            Focal loss.
        """
        # Convert inputs to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss function."""
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        margin: float = 0.3,
        reduction: str = "mean"
    ):
        """
        Initialize combined loss.
        
        Args:
            focal_gamma: Focal loss gamma parameter.
            focal_alpha: Focal loss alpha parameter.
            margin: Margin for ranking loss.
            reduction: Reduction method.
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction=reduction)
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs: Input logits.
            targets: Target labels.
            pos_mask: Positive sample mask.
            neg_mask: Negative sample mask.
            
        Returns:
            Tuple of (loss, loss_components).
        """
        # Calculate focal loss
        focal_loss = self.focal_loss(inputs, targets)
        
        # Calculate ranking loss if masks are provided
        if pos_mask is not None and neg_mask is not None:
            # Extract positive and negative scores
            pos_scores = inputs[pos_mask]
            neg_scores = inputs[neg_mask]
            
            # Calculate all pairs of positive-negative scores
            pos_scores = pos_scores.unsqueeze(1)  # [num_pos, 1]
            neg_scores = neg_scores.unsqueeze(0)  # [1, num_neg]
            
            # Calculate margin ranking loss
            margin_diff = self.margin - pos_scores + neg_scores  # [num_pos, num_neg]
            rank_loss = F.relu(margin_diff)
            
            # Apply reduction
            if self.reduction == "mean":
                rank_loss = rank_loss.mean()
            elif self.reduction == "sum":
                rank_loss = rank_loss.sum()
            
            # Combine losses
            loss = focal_loss + rank_loss
            loss_components = {"focal_loss": focal_loss, "rank_loss": rank_loss}
        else:
            # Only focal loss
            loss = focal_loss
            loss_components = {"focal_loss": focal_loss}
        
        return loss, loss_components
