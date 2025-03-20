"""
Configuration settings for the TCM target prioritization system.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

@dataclass
class DataConfig:
    """Data configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    # Knowledge graph configuration
    entity_types: List[str] = None
    relation_types: List[str] = None
    # Data sources
    drug_target_sources: List[str] = None
    disease_target_sources: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["tcm", "compound", "target", "disease"]
        if self.relation_types is None:
            self.relation_types = [
                "contains", "binds", "inhibits", "activates", 
                "associated_with", "linked_with"
            ]
        if self.drug_target_sources is None:
            self.drug_target_sources = [
                "BindingDB", "DrugBank", "TCMSP", "PubChem"
            ]
        if self.disease_target_sources is None:
            self.disease_target_sources = [
                "OMIM", "DisGeNET", "TTD"
            ]

@dataclass
class FeatureConfig:
    """Feature configuration."""
    # Feature dimensions
    compound_feature_dim: int = 1024
    target_feature_dim: int = 1024
    disease_feature_dim: int = 512
    unified_feature_dim: int = 256
    # Compound feature configuration
    use_morgan_fingerprint: bool = True
    morgan_radius: int = 2
    morgan_nbits: int = 1024
    # Target feature configuration
    use_sequence_encoding: bool = True
    use_blosum: bool = True
    # Disease feature configuration
    use_ontology: bool = True
    # Feature normalization
    normalize_features: bool = True

@dataclass
class ModelConfig:
    """Model configuration."""
    # Model architecture
    num_layers: int = 3
    hidden_dim: int = 256
    dropout: float = 0.3
    residual: bool = True
    layer_norm: bool = True
    # Attention mechanism
    use_attention: bool = True
    num_heads: int = 4
    attention_dropout: float = 0.1
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    # Optimizer
    optimizer: str = "adam"
    # Loss function
    loss_fn: str = "combined"
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

@dataclass
class RankingConfig:
    """Ranking configuration."""
    # Ranking weights
    similarity_weight: float = 0.4  # ω1
    disease_weight: float = 0.3     # ω2
    centrality_weight: float = 0.3  # ω3
    # Weight optimization
    optimize_weights: bool = True
    weight_grid_search_step: float = 0.1
    weight_cv_folds: int = 5
    # Centrality measure
    centrality_method: str = "pagerank"
    # Ranking threshold
    confidence_threshold: float = 0.5

@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    ranking: RankingConfig = RankingConfig()
    # General configuration
    random_seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": vars(self.data),
            "features": vars(self.features),
            "model": vars(self.model),
            "ranking": vars(self.ranking),
            "random_seed": self.random_seed,
            "device": self.device,
            "log_dir": self.log_dir,
            "checkpoint_dir": self.checkpoint_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        feature_config = FeatureConfig(**config_dict.get("features", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        ranking_config = RankingConfig(**config_dict.get("ranking", {}))
        
        return cls(
            data=data_config,
            features=feature_config,
            model=model_config,
            ranking=ranking_config,
            random_seed=config_dict.get("random_seed", 42),
            device=config_dict.get("device", "cuda"),
            log_dir=config_dict.get("log_dir", "logs"),
            checkpoint_dir=config_dict.get("checkpoint_dir", "checkpoints")
        )
