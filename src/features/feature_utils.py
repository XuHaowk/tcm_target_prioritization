"""
Feature utilities for TCM target prioritization system.
"""
import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.config import FeatureConfig
from src.data.knowledge_graph import KnowledgeGraph
from src.features.compound_features import CompoundFeatureExtractor, TCMCompoundFeatureExtractor
from src.features.target_features import TargetFeatureExtractor
from src.features.disease_features import DiseaseFeatureExtractor

class FeatureBuilder:
    """Feature builder for TCM target prioritization."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature builder.
        
        Args:
            config: Feature configuration.
        """
        self.config = config
        
        # Feature extractors
        self.compound_extractor = TCMCompoundFeatureExtractor(config)
        self.target_extractor = TargetFeatureExtractor(config)
        self.disease_extractor = DiseaseFeatureExtractor(config)
        
        # Dimensionality reduction
        self.use_dimensionality_reduction = False
        self.compound_pca = None
        self.target_pca = None
        self.disease_pca = None
        
        # Feature normalization
        self.normalize_features = config.normalize_features
        self.compound_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.disease_scaler = StandardScaler()
    
    def enable_dimensionality_reduction(self, target_dim: int = 256) -> None:
        """
        Enable dimensionality reduction.
        
        Args:
            target_dim: Target dimension.
        """
        self.use_dimensionality_reduction = True
        self.compound_pca = PCA(n_components=target_dim)
        self.target_pca = PCA(n_components=target_dim)
        self.disease_pca = PCA(n_components=target_dim)
    
    def build_compound_features(
        self,
        smiles_mapping: Dict[str, str],
        tcm_metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Build compound features.
        
        Args:
            smiles_mapping: Dictionary mapping compound IDs to SMILES strings.
            tcm_metadata: TCM metadata.
            
        Returns:
            Dictionary mapping compound IDs to features.
        """
        # Load TCM metadata if provided
        if tcm_metadata is not None:
            self.compound_extractor.load_tcm_metadata(tcm_metadata)
        
        # Extract features
        features = self.compound_extractor.extract_features_from_mapping(smiles_mapping)
        
        # Convert to matrix for normalization and dimensionality reduction
        compound_ids = list(features.keys())
        feature_matrix = np.stack([features[cid] for cid in compound_ids])
        
        # Normalize features
        if self.normalize_features:
            feature_matrix = self.compound_scaler.fit_transform(feature_matrix)
        
        # Apply dimensionality reduction if enabled
        if self.use_dimensionality_reduction:
            feature_matrix = self.compound_pca.fit_transform(feature_matrix)
        
        # Convert back to dictionary
        result = {}
        for i, compound_id in enumerate(compound_ids):
            result[compound_id] = feature_matrix[i]
        
        return result
    
    def build_target_features(
        self,
        sequence_mapping: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """
        Build target features.
        
        Args:
            sequence_mapping: Dictionary mapping target IDs to protein sequences.
            
        Returns:
            Dictionary mapping target IDs to features.
        """
        # Extract features
        features = self.target_extractor.extract_features_from_mapping(sequence_mapping)
        
        # Convert to matrix for normalization and dimensionality reduction
        target_ids = list(features.keys())
        feature_matrix = np.stack([features[tid] for tid in target_ids])
        
        # Normalize features
        if self.normalize_features:
            feature_matrix = self.target_scaler.fit_transform(feature_matrix)
        
        # Apply dimensionality reduction if enabled
        if self.use_dimensionality_reduction:
            feature_matrix = self.target_pca.fit_transform(feature_matrix)
        
        # Convert back to dictionary
        result = {}
        for i, target_id in enumerate(target_ids):
            result[target_id] = feature_matrix[i]
        
        return result
    
    def build_disease_features(
        self,
        ontology_data: Dict[str, Dict[str, str]],
        kg_data: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Build disease features.
        
        Args:
            ontology_data: Disease ontology data.
            kg_data: Knowledge graph data.
            
        Returns:
            Dictionary mapping disease IDs to features.
        """
        # Extract features
        features = self.disease_extractor.extract_features_from_mapping(ontology_data, kg_data)
        
        # Convert to matrix for normalization and dimensionality reduction
        disease_ids = list(features.keys())
        feature_matrix = np.stack([features[did] for did in disease_ids])
        
        # Normalize features
        if self.normalize_features:
            feature_matrix = self.disease_scaler.fit_transform(feature_matrix)
        
        # Apply dimensionality reduction if enabled
        if self.use_dimensionality_reduction:
            feature_matrix = self.disease_pca.fit_transform(feature_matrix)
        
        # Convert back to dictionary
        result = {}
        for i, disease_id in enumerate(disease_ids):
            result[disease_id] = feature_matrix[i]
        
        return result
    
    def build_node_features(
        self,
        kg: KnowledgeGraph,
        smiles_mapping: Dict[str, str],
        sequence_mapping: Dict[str, str],
        ontology_data: Dict[str, Dict[str, str]],
        tcm_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        kg_data: Optional[Dict[str, List[str]]] = None,
        enforce_unified_dim: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Build node features for all entity types in the knowledge graph.
        
        Args:
            kg: Knowledge graph.
            smiles_mapping: Dictionary mapping compound IDs to SMILES strings.
            sequence_mapping: Dictionary mapping target IDs to protein sequences.
            ontology_data: Disease ontology data.
            tcm_metadata: TCM metadata.
            kg_data: Knowledge graph data.
            enforce_unified_dim: Whether to enforce unified dimension for all features.
            
        Returns:
            Dictionary mapping entity types to feature tensors.
        """
        # Build features for each entity type
        compound_features = self.build_compound_features(smiles_mapping, tcm_metadata)
        target_features = self.build_target_features(sequence_mapping)
        disease_features = self.build_disease_features(ontology_data, kg_data)
        
        # Create TCM features (use average of contained compounds)
        tcm_features = {}
        for tcm_id in kg.entity_to_idx["tcm"]:
            # Get neighbors of TCM
            neighbors = kg.get_entity_neighbors(tcm_id, "tcm")
            
            # Extract compound neighbors
            compounds = []
            for relation, neighbor_list in neighbors.items():
                if relation == "contains":
                    for neighbor_id, neighbor_type, _ in neighbor_list:
                        if neighbor_type == "compound" and neighbor_id in compound_features:
                            compounds.append(neighbor_id)
            
            # Average compound features
            if compounds:
                tcm_feature = np.mean([compound_features[cid] for cid in compounds], axis=0)
            else:
                # Use default features
                tcm_feature = np.zeros(self.config.unified_feature_dim)
            
            tcm_features[tcm_id] = tcm_feature
        
        # Ensure all features have the same dimension if required
        if enforce_unified_dim:
            target_dim = self.config.unified_feature_dim
            
            # Dimensionality reduction or padding for compound features
            if list(compound_features.values())[0].shape[0] != target_dim:
                compound_features = self._resize_features(compound_features, target_dim)
            
            # Dimensionality reduction or padding for target features
            if list(target_features.values())[0].shape[0] != target_dim:
                target_features = self._resize_features(target_features, target_dim)
            
            # Dimensionality reduction or padding for disease features
            if list(disease_features.values())[0].shape[0] != target_dim:
                disease_features = self._resize_features(disease_features, target_dim)
            
            # Dimensionality reduction or padding for TCM features
            if list(tcm_features.values())[0].shape[0] != target_dim:
                tcm_features = self._resize_features(tcm_features, target_dim)
        
        # Convert to tensors
        node_features = {}
        
        # Compound features
        compound_tensor = torch.zeros(kg.entity_counts["compound"], list(compound_features.values())[0].shape[0])
        for compound_id, idx in kg.entity_to_idx["compound"].items():
            if compound_id in compound_features:
                compound_tensor[idx] = torch.tensor(compound_features[compound_id])
        node_features["compound"] = compound_tensor
        
        # Target features
        target_tensor = torch.zeros(kg.entity_counts["target"], list(target_features.values())[0].shape[0])
        for target_id, idx in kg.entity_to_idx["target"].items():
            if target_id in target_features:
                target_tensor[idx] = torch.tensor(target_features[target_id])
        node_features["target"] = target_tensor
        
        # Disease features
        disease_tensor = torch.zeros(kg.entity_counts["disease"], list(disease_features.values())[0].shape[0])
        for disease_id, idx in kg.entity_to_idx["disease"].items():
            if disease_id in disease_features:
                disease_tensor[idx] = torch.tensor(disease_features[disease_id])
        node_features["disease"] = disease_tensor
        
        # TCM features
        tcm_tensor = torch.zeros(kg.entity_counts["tcm"], list(tcm_features.values())[0].shape[0])
        for tcm_id, idx in kg.entity_to_idx["tcm"].items():
            if tcm_id in tcm_features:
                tcm_tensor[idx] = torch.tensor(tcm_features[tcm_id])
        node_features["tcm"] = tcm_tensor
        
        return node_features
    
    def _resize_features(self, feature_matrix, target_dim):
        """Resize feature matrix to target dimension."""
        n_samples, n_features = feature_matrix.shape
    
        # If we have fewer samples than target_dim, we can't use PCA to increase dimensions
        if target_dim > min(n_samples, n_features):
            print(f"Warning: Cannot use PCA to increase dimensions. Requested {target_dim} but limited to {min(n_samples, n_features)}.")
        
            # Option 1: Use original features with padding
            if n_features <= min(n_samples, n_features):
                padded_features = np.zeros((n_samples, min(n_samples, n_features)))
                padded_features[:, :n_features] = feature_matrix
                return padded_features
            
            # Option 2: Use PCA to reduce to maximum possible dimensions
            target_dim = min(n_samples, n_features)
    
        # Standard PCA for dimensionality reduction
        if n_features > target_dim:
            pca = PCA(n_components=target_dim)
            feature_matrix = pca.fit_transform(feature_matrix)
    
        return feature_matrix
    
    def save(self, file_path: str) -> None:
        """
        Save feature builder.
        
        Args:
            file_path: Path to the output file.
        """
        import pickle
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str) -> "FeatureBuilder":
        """
        Load feature builder.
        
        Args:
            file_path: Path to the input file.
            
        Returns:
            Loaded feature builder.
        """
        import pickle
        
        with open(file_path, "rb") as f:
            return pickle.load(f)

