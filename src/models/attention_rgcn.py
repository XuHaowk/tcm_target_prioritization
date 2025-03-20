"""
Attention-enhanced Relational Graph Convolutional Network for TCM target prioritization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Dict, List, Tuple, Optional, Union, Set
from src.config import ModelConfig
from src.models.layers import AttentionLayer, LayerNorm

class AttentionRGCN(nn.Module):
    """Attention-enhanced Relational Graph Convolutional Network."""
    
    def __init__(
        self,
        entity_types: List[str],
        relation_types: List[str],
        feature_dims: Dict[str, int],
        config: ModelConfig
    ):
        """
        Initialize AttentionRGCN.
        
        Args:
            entity_types: List of entity types.
            relation_types: List of relation types.
            feature_dims: Dictionary mapping entity types to feature dimensions.
            config: Model configuration.
        """
        super().__init__()
        
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.feature_dims = feature_dims
        self.config = config
        
        # Model dimensions
        self.num_relations = len(relation_types)
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.use_attention = config.use_attention
        self.num_heads = config.num_heads
        self.residual = config.residual
        self.layer_norm = config.layer_norm
        
        # Input feature projections
        self.input_projections = nn.ModuleDict()
        for entity_type, feature_dim in feature_dims.items():
            self.input_projections[entity_type] = nn.Linear(
                feature_dim, 
                self.hidden_dim
            )
        
        # RGCN layers
        self.rgcn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.rgcn_layers.append(
                RGCNConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_relations=self.num_relations,
                    num_bases=None,
                    aggr="add"
                )
            )
        
        # Attention layers
        if self.use_attention:
            self.attention_layers = nn.ModuleList()
            for i in range(self.num_layers):
                self.attention_layers.append(
                    AttentionLayer(
                        hidden_dim=self.hidden_dim,
                        num_heads=self.num_heads,
                        dropout=config.attention_dropout
                    )
                )
        
        # Layer normalization
        if self.layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(self.num_layers):
                self.layer_norms.append(
                    LayerNorm(self.hidden_dim)
                )
        
        # Output layers
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_features: Dictionary mapping entity types to node features.
            edge_index_dict: Dictionary mapping (src_type, rel_type, dst_type) to edge indices.
            edge_attr_dict: Dictionary mapping (src_type, rel_type, dst_type) to edge attributes.
            
        Returns:
            Dictionary mapping entity types to output node embeddings.
        """
        # Project input features
        hidden_features = {}
        for entity_type, features in node_features.items():
            if entity_type in self.input_projections:
                hidden_features[entity_type] = self.input_projections[entity_type](features)
            else:
                # For entity types without input projections, use zero embeddings
                hidden_features[entity_type] = torch.zeros(
                    features.size(0), 
                    self.hidden_dim, 
                    device=features.device
                )
        
        # Create relation type to index mapping
        rel_type_to_idx = {rel: idx for idx, rel in enumerate(self.relation_types)}
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            # Prepare edge index and type for RGCN
            edge_indices = []
            edge_types = []
            
            for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
                # Skip edges involving entity types not in feature_dims
                if src_type not in hidden_features or dst_type not in hidden_features:
                    continue
                
                # Get relation index
                rel_idx = rel_type_to_idx[rel_type]
                
                # Append to lists
                edge_indices.append(edge_index)
                edge_types.append(torch.full((edge_index.size(1),), rel_idx, dtype=torch.long))
            
            # Concatenate edge indices and types
            if edge_indices:
                combined_edge_index = torch.cat(edge_indices, dim=1)
                combined_edge_type = torch.cat(edge_types, dim=0)
            else:
                # No edges, use dummy values
                combined_edge_index = torch.zeros((2, 0), dtype=torch.long)
                combined_edge_type = torch.zeros((0,), dtype=torch.long)
            
            # Combine node features
            node_feature_list = []
            node_type_indices = {}
            offset = 0
            
            for entity_type, features in hidden_features.items():
                node_feature_list.append(features)
                node_type_indices[entity_type] = (offset, offset + features.size(0))
                offset += features.size(0)
            
            combined_node_features = torch.cat(node_feature_list, dim=0)
            
            # Apply RGCN layer
            combined_output = self.rgcn_layers[layer_idx](
                combined_node_features,
                combined_edge_index,
                combined_edge_type
            )
            
            # Apply attention if configured
            if self.use_attention:
                combined_output = self.attention_layers[layer_idx](
                    combined_output,
                    combined_edge_index,
                    combined_edge_type
                )
            
            # Split output by entity type
            layer_output = {}
            for entity_type, (start, end) in node_type_indices.items():
                layer_output[entity_type] = combined_output[start:end]
            
            # Apply residual connection if configured
            if self.residual and layer_idx > 0:
                for entity_type in layer_output:
                    layer_output[entity_type] = layer_output[entity_type] + hidden_features[entity_type]
            
            # Apply layer normalization if configured
            if self.layer_norm:
                for entity_type in layer_output:
                    layer_output[entity_type] = self.layer_norms[layer_idx](layer_output[entity_type])
            
            # Apply dropout
            for entity_type in layer_output:
                layer_output[entity_type] = F.dropout(
                    layer_output[entity_type],
                    p=self.dropout,
                    training=self.training
                )
            
            # Update hidden features
            hidden_features = layer_output
        
        # Apply output projection
        output = {}
        for entity_type, features in hidden_features.items():
            output[entity_type] = self.output_projection(features)
        
        return output
    
    def predict_link(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probability between source and target entities.
        
        Args:
            source_embeds: Source entity embeddings.
            target_embeds: Target entity embeddings.
            
        Returns:
            Link probabilities.
        """
        # Calculate cosine similarity
        source_norm = F.normalize(source_embeds, p=2, dim=1)
        target_norm = F.normalize(target_embeds, p=2, dim=1)
        
        # Calculate similarity scores
        scores = torch.mm(source_norm, target_norm.t())
        
        # Convert to probabilities
        probabilities = torch.sigmoid(scores)
        
        return probabilities
    
    def predict_compound_target_links(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        compound_indices: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict links between compounds and targets.
        
        Args:
            node_embeddings: Dictionary mapping entity types to node embeddings.
            compound_indices: Compound indices.
            target_indices: Target indices.
            
        Returns:
            Link probabilities.
        """
        # Get compound and target embeddings
        compound_embeds = node_embeddings["compound"][compound_indices]
        target_embeds = node_embeddings["target"][target_indices]
        
        # Predict links
        return self.predict_link(compound_embeds, target_embeds)
    
    def save(self, file_path: str) -> None:
        """
        Save model.
        
        Args:
            file_path: Path to the output file.
        """
        torch.save(
            {
                "entity_types": self.entity_types,
                "relation_types": self.relation_types,
                "feature_dims": self.feature_dims,
                "config": self.config,
                "state_dict": self.state_dict()
            },
            file_path
        )
    
    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "AttentionRGCN":
        """
        Load model.
        
        Args:
            file_path: Path to the input file.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
        """
        checkpoint = torch.load(file_path, map_location=device)
        
        model = cls(
            entity_types=checkpoint["entity_types"],
            relation_types=checkpoint["relation_types"],
            feature_dims=checkpoint["feature_dims"],
            config=checkpoint["config"]
        )
        
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        return model
