"""
Data processing utilities for TCM target prioritization system.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from src.config import DataConfig
from src.data.knowledge_graph import KnowledgeGraph

class GraphDataset(Dataset):
    """Graph dataset for TCM target prioritization."""
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        pairs: List[Tuple[str, str]],
        labels: List[float],
        node_features: Dict[str, torch.Tensor]
    ):
        """
        Initialize graph dataset.
        
        Args:
            kg: Knowledge graph.
            pairs: List of (compound_id, target_id) pairs.
            labels: List of labels.
            node_features: Node features for each entity type.
        """
        self.kg = kg
        self.pairs = pairs
        self.labels = labels
        self.node_features = node_features
        
        # Validate pairs
        valid_pairs = []
        valid_labels = []
        for (compound_id, target_id), label in zip(pairs, labels):
            if (compound_id in kg.entity_to_idx["compound"] and
                target_id in kg.entity_to_idx["target"]):
                valid_pairs.append((compound_id, target_id))
                valid_labels.append(label)
        
        self.valid_pairs = valid_pairs
        self.valid_labels = valid_labels
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], float]:
        """
        Get dataset item.
        
        Args:
            idx: Item index.
            
        Returns:
            Tuple of (compound_id, target_id) pair and label.
        """
        return self.valid_pairs[idx], self.valid_labels[idx]
    
    def get_node_features(self, entity_id: str, entity_type: str) -> torch.Tensor:
        """
        Get node features for an entity.
        
        Args:
            entity_id: Entity identifier.
            entity_type: Entity type.
            
        Returns:
            Node features tensor.
        """
        if entity_type not in self.kg.entity_to_idx:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        if entity_id not in self.kg.entity_to_idx[entity_type]:
            raise ValueError(f"Unknown entity: {entity_id}")
        
        entity_idx = self.kg.entity_to_idx[entity_type][entity_id]
        return self.node_features[entity_type][entity_idx]
    
    def get_subgraph(
        self,
        idx: int,
        max_hops: int = 2
    ) -> Tuple[KnowledgeGraph, Dict[str, torch.Tensor]]:
        """
        Get subgraph around a compound-target pair.
        
        Args:
            idx: Item index.
            max_hops: Maximum number of hops from the entities.
            
        Returns:
            Tuple of subgraph and node features.
        """
        compound_id, target_id = self.valid_pairs[idx]
        
        # Get subgraph
        subgraph = self.kg.get_entity_subgraph(
            entity_ids=[compound_id, target_id],
            entity_types=["compound", "target"],
            max_hops=max_hops
        )
        
        # Extract node features for subgraph
        subgraph_features = {}
        for entity_type in self.kg.entity_types:
            if entity_type in subgraph.entity_counts and subgraph.entity_counts[entity_type] > 0:
                features = []
                for idx in range(subgraph.entity_counts[entity_type]):
                    entity_id = subgraph.idx_to_entity[entity_type][idx]
                    original_idx = self.kg.entity_to_idx[entity_type][entity_id]
                    features.append(self.node_features[entity_type][original_idx])
                
                subgraph_features[entity_type] = torch.stack(features)
        
        return subgraph, subgraph_features
    
    def to_train_batch(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        max_hops: int = 2
    ) -> List[Tuple[KnowledgeGraph, Dict[str, torch.Tensor], float]]:
        """
        Convert dataset to batch of training samples.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            max_hops: Maximum number of hops for subgraphs.
            
        Returns:
            List of (subgraph, node_features, label) tuples.
        """
        indices = list(range(len(self)))
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = []
            
            for idx in batch_indices:
                subgraph, subgraph_features = self.get_subgraph(idx, max_hops)
                label = self.valid_labels[idx]
                batch_data.append((subgraph, subgraph_features, label))
            
            batches.append(batch_data)
        
        return batches

class DataProcessor:
    """Data processor for TCM target prioritization."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data processor.
        
        Args:
            config: Data configuration.
        """
        self.config = config
    
    def process_drug_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process drug-target data.
        
        Args:
            df: Drug-target data.
            
        Returns:
            Processed drug-target data.
        """
        # Standardize column names
        required_columns = ["compound_id", "target_id"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Standardize relation types
        if "relation" in df.columns:
            # Map relation types to standard types
            relation_mapping = {
                "inhibits": "inhibits",
                "inhibitor": "inhibits",
                "inhibition": "inhibits",
                "antagonist": "inhibits",
                "activates": "activates",
                "activator": "activates",
                "activation": "activates",
                "agonist": "activates",
                "binds": "binds",
                "binding": "binds",
                "binder": "binds",
                "substrate": "binds",
                "associated_with": "associated_with",
                "associated": "associated_with",
                "association": "associated_with",
                "related": "associated_with",
                "linked_with": "linked_with",
                "linked": "linked_with",
                "link": "linked_with"
            }
            
            df["relation"] = df["relation"].str.lower().map(
                lambda x: relation_mapping.get(x, "binds")
            )
        else:
            # Default relation
            df["relation"] = "binds"
        
        # Add confidence score if not present
        if "confidence" not in df.columns:
            df["confidence"] = 1.0
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["compound_id", "target_id", "relation"])
        
        return df
    
    def process_disease_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process disease-target data.
        
        Args:
            df: Disease-target data.
            
        Returns:
            Processed disease-target data.
        """
        # Standardize column names
        required_columns = ["disease_id", "target_id"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Standardize relation types
        if "relation" in df.columns:
            # Map relation types to standard types
            relation_mapping = {
                "associated_with": "associated_with",
                "associated": "associated_with",
                "association": "associated_with",
                "related": "associated_with",
                "linked_with": "linked_with",
                "linked": "linked_with",
                "link": "linked_with",
                "causal": "causal",
                "causes": "causal",
                "caused_by": "causal",
                "therapeutic": "therapeutic",
                "therapeutic_target": "therapeutic",
                "treats": "therapeutic"
            }
            
            df["relation"] = df["relation"].str.lower().map(
                lambda x: relation_mapping.get(x, "associated_with")
            )
        else:
            # Default relation
            df["relation"] = "associated_with"
        
        # Add confidence score if not present
        if "confidence" not in df.columns:
            df["confidence"] = 1.0
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["disease_id", "target_id", "relation"])
        
        return df
    
    def process_tcm_compound_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process TCM compound data.
        
        Args:
            df: TCM compound data.
            
        Returns:
            Processed TCM compound data.
        """
        # Standardize column names
        required_columns = ["tcm_id", "compound_id"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Add confidence score if not present
        if "confidence" not in df.columns:
            df["confidence"] = 1.0
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["tcm_id", "compound_id"])
        
        return df
    
    def generate_negative_samples(
        self,
        kg: KnowledgeGraph,
        positive_pairs: List[Tuple[str, str]],
        negative_ratio: float = 1.0
    ) -> List[Tuple[str, str]]:
        """
        Generate negative samples.
        
        Args:
            kg: Knowledge graph.
            positive_pairs: List of positive (compound_id, target_id) pairs.
            negative_ratio: Ratio of negative to positive samples.
            
        Returns:
            List of negative (compound_id, target_id) pairs.
        """
        # Set of positive pairs for fast lookup
        positive_set = set(positive_pairs)
        
        # Set of all compounds and targets
        compounds = set(kg.entity_to_idx["compound"].keys())
        targets = set(kg.entity_to_idx["target"].keys())
        
        # Generate negative samples
        num_negatives = int(len(positive_pairs) * negative_ratio)
        negative_pairs = []
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(negative_pairs) < num_negatives and attempts < max_attempts:
            compound_id = np.random.choice(list(compounds))
            target_id = np.random.choice(list(targets))
            
            pair = (compound_id, target_id)
            if pair not in positive_set and pair not in negative_pairs:
                negative_pairs.append(pair)
            
            attempts += 1
        
        return negative_pairs
    
    def create_cross_validation_folds(
        self,
        pairs: List[Tuple[str, str]],
        labels: List[float],
        n_folds: int = 5,
        stratify: bool = True
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create cross-validation folds.
        
        Args:
            pairs: List of (compound_id, target_id) pairs.
            labels: List of labels.
            n_folds: Number of folds.
            stratify: Whether to stratify by labels.
            
        Returns:
            List of (train_indices, val_indices) tuples.
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        if stratify:
            folder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(folder.split(pairs, labels))
        else:
            folder = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(folder.split(pairs))
        
        return splits
    
    def leave_one_out(
        self,
        pairs: List[Tuple[str, str]],
        labels: List[float],
        holdout_compounds: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Create leave-one-out split.
        
        Args:
            pairs: List of (compound_id, target_id) pairs.
            labels: List of labels.
            holdout_compounds: List of compound IDs to hold out.
            
        Returns:
            Tuple of (train_indices, test_indices).
        """
        train_indices = []
        test_indices = []
        
        for i, (compound_id, _) in enumerate(pairs):
            if compound_id in holdout_compounds:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        return train_indices, test_indices
    
    def create_train_val_test_split(
        self,
        pairs: List[Tuple[str, str]],
        labels: List[float],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create train-validation-test split.
        
        Args:
            pairs: List of (compound_id, target_id) pairs.
            labels: List of labels.
            val_ratio: Validation ratio.
            test_ratio: Test ratio.
            stratify: Whether to stratify by labels.
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        from sklearn.model_selection import train_test_split
        
        # First, split off test set
        train_val_indices, test_indices = train_test_split(
            np.arange(len(pairs)),
            test_size=test_ratio,
            random_state=42,
            stratify=labels if stratify else None
        )
        
        # Then, split remaining data into train and validation sets
        train_val_labels = [labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio / (1 - test_ratio),
            random_state=42,
            stratify=train_val_labels if stratify else None
        )
        
        return train_indices, val_indices, test_indices
    
    def create_graph_dataset(
        self,
        kg: KnowledgeGraph,
        pairs: List[Tuple[str, str]],
        labels: List[float],
        node_features: Dict[str, torch.Tensor]
    ) -> GraphDataset:
        """
        Create graph dataset.
        
        Args:
            kg: Knowledge graph.
            pairs: List of (compound_id, target_id) pairs.
            labels: List of labels.
            node_features: Node features for each entity type.
            
        Returns:
            Graph dataset.
        """
        return GraphDataset(kg, pairs, labels, node_features)
    
    def split_dataset(
        self,
        dataset: GraphDataset,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int]
    ) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Graph dataset.
            train_indices: Training indices.
            val_indices: Validation indices.
            test_indices: Test indices.
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        def subset_dataset(dataset, indices):
            pairs = [dataset.valid_pairs[i] for i in indices]
            labels = [dataset.valid_labels[i] for i in indices]
            return GraphDataset(dataset.kg, pairs, labels, dataset.node_features)
        
        train_dataset = subset_dataset(dataset, train_indices)
        val_dataset = subset_dataset(dataset, val_indices)
        test_dataset = subset_dataset(dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
