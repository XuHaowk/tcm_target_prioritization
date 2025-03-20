"""
Target ranking for TCM target prioritization system.
"""
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from src.config import RankingConfig
from src.data.knowledge_graph import KnowledgeGraph
from src.models.attention_rgcn import AttentionRGCN

class TargetRanker:
    """Target ranker for TCM target prioritization."""
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        model: AttentionRGCN,
        config: RankingConfig,
        device: str = "cpu",
        node_features: Optional[Dict[str, torch.Tensor]] = None  # Add this parameter
    ):
        """Initialize target ranker."""
        self.kg = kg
        self.model = model
        self.config = config
        self.device = device
        self.node_features = node_features  # Store node features
    
        # Move model to device
        self.model.to(self.device)
    
        # Set model to evaluation mode
        self.model.eval()
    
        # Ranking weights
        self.similarity_weight = config.similarity_weight
        self.disease_weight = config.disease_weight
        self.centrality_weight = config.centrality_weight
    
        # Pre-compute node centrality
        self.node_centrality = self.kg.calculate_centrality(
            centrality_method=config.centrality_method
        )
    
    def predict_score(
        self,
        compound_id: str,
        target_id: str,
        node_embeddings: Dict[str, torch.Tensor]
    ) -> float:
        """
        Predict score for compound-target pair.
        
        Args:
            compound_id: Compound identifier.
            target_id: Target identifier.
            node_embeddings: Node embeddings.
            
        Returns:
            Prediction score.
        """
        # Get compound and target indices
        compound_idx = self.kg.entity_to_idx["compound"][compound_id]
        target_idx = self.kg.entity_to_idx["target"][target_id]
        
        # Get embeddings
        compound_embed = node_embeddings["compound"][compound_idx].unsqueeze(0)
        target_embed = node_embeddings["target"][target_idx].unsqueeze(0)
        
        # Calculate similarity score using cosine similarity
        with torch.no_grad():
            compound_norm = torch.nn.functional.normalize(compound_embed, p=2, dim=1)
            target_norm = torch.nn.functional.normalize(target_embed, p=2, dim=1)
            similarity = torch.mm(compound_norm, target_norm.t()).item()
        
        return similarity
    
    def calculate_disease_priority(
        self,
        target_id: str,
        disease_id: Optional[str] = None
    ) -> float:
        """
        Calculate disease priority score for target.
        
        Args:
            target_id: Target identifier.
            disease_id: Disease identifier. If None, return average across all diseases.
            
        Returns:
            Disease priority score.
        """
        # Get target index
        target_idx = self.kg.entity_to_idx["target"][target_id]
        target_node = ("target", target_idx)
        
        # Check if disease_id is provided
        if disease_id is not None:
            # Get disease index
            if disease_id not in self.kg.entity_to_idx["disease"]:
                return 0.0
            
            disease_idx = self.kg.entity_to_idx["disease"][disease_id]
            disease_node = ("disease", disease_idx)
            
            # Calculate shortest path length
            try:
                path_length = nx.shortest_path_length(
                    self.kg.nx_graph,
                    source=target_node,
                    target=disease_node
                )
                
                # Convert to priority score (higher is better)
                priority_score = 1.0 / (path_length + 1)
            except nx.NetworkXNoPath:
                # No path exists
                priority_score = 0.0
        else:
            # Calculate average priority across all diseases
            priority_scores = []
            for disease_id in self.kg.entity_to_idx["disease"]:
                disease_idx = self.kg.entity_to_idx["disease"][disease_id]
                disease_node = ("disease", disease_idx)
                
                try:
                    path_length = nx.shortest_path_length(
                        self.kg.nx_graph,
                        source=target_node,
                        target=disease_node
                    )
                    
                    # Convert to priority score (higher is better)
                    priority_score = 1.0 / (path_length + 1)
                    priority_scores.append(priority_score)
                except nx.NetworkXNoPath:
                    # No path exists
                    priority_scores.append(0.0)
            
            # Calculate average priority
            if priority_scores:
                priority_score = np.mean(priority_scores)
            else:
                priority_score = 0.0
        
        return priority_score
    
    def calculate_centrality(self, target_id: str) -> float:
        """
        Calculate centrality score for target.
        
        Args:
            target_id: Target identifier.
            
        Returns:
            Centrality score.
        """
        # Get target index
        target_idx = self.kg.entity_to_idx["target"][target_id]
        target_node = ("target", target_idx)
        
        # Get centrality score
        centrality_score = self.node_centrality.get(target_node, 0.0)
        
        return centrality_score
    
        # Get all targets
        targets = list(self.kg.entity_to_idx["target"].keys())
        
        # Calculate scores for each target
        scores = []
        for target_id in targets:
            # Calculate similarity score
            similarity = self.predict_score(compound_id, target_id, node_embeddings)
            
            # Calculate disease priority
            disease_priority = self.calculate_disease_priority(target_id, disease_id)
            
            # Calculate centrality
            centrality = self.calculate_centrality(target_id)
            
            # Calculate weighted score
            weighted_score = (
                self.similarity_weight * similarity +
                self.disease_weight * disease_priority +
                self.centrality_weight * centrality
            )
            
            # Add to scores
            scores.append({
                "target_id": target_id,
                "similarity": similarity,
                "disease_priority": disease_priority,
                "centrality": centrality,
                "weighted_score": weighted_score,
                "rank": 0
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(scores)
        
        # Sort by weighted score
        df = df.sort_values("weighted_score", ascending=False)
        
        # Add rank
        df["rank"] = range(1, len(df) + 1)
        
        # Filter by confidence threshold
        df = df[df["weighted_score"] >= confidence_threshold]
        
        # Select top_k targets if specified
        if top_k is not None and top_k > 0:
            df = df.head(top_k)
        
        return df
    
    def rank_targets(
        self,
        compound_id: str,
        disease_id: Optional[str] = None,
        node_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Rank targets for compound."""
        # Check if compound exists
        if compound_id not in self.kg.entity_to_idx["compound"]:
            raise ValueError(f"Compound not found: {compound_id}")
    
        # Use config values if not provided
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
    
        # Compute node embeddings if not provided
        if node_embeddings is None:
            # Convert knowledge graph to PyTorch Geometric format
            pyg_data = self.kg.to_torch_geometric()
        
            # Extract node features and move to device
            node_features = {}
            if self.node_features is not None:
                # Use pre-computed node features if available
                node_features = {k: v.to(self.device) for k, v in self.node_features.items()}
            else:
                # Otherwise extract from PyG data
                for entity_type in self.kg.entity_types:
                    if entity_type in pyg_data and hasattr(pyg_data[entity_type], 'x'):
                        node_features[entity_type] = pyg_data[entity_type].x.to(self.device)
        
            # Extract edge indices and attributes
            edge_index_dict = {}
            edge_attr_dict = {}
        
            for edge_type, edge_info in pyg_data.edge_items():
                edge_index_dict[edge_type] = edge_info.edge_index.to(self.device)
                edge_attr_dict[edge_type] = edge_info.edge_attr.to(self.device)
        
            # Compute node embeddings
            with torch.no_grad():
                node_embeddings = self.model(node_features, edge_index_dict, edge_attr_dict)

        
        # Rank targets for each compound
        for compound_id in compound_ids:
            try:
                results[compound_id] = self.rank_targets(
                    compound_id=compound_id,
                    disease_id=disease_id,
                    node_embeddings=node_embeddings,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold
                )
            except ValueError:
                # Compound not found
                results[compound_id] = pd.DataFrame(columns=[
                    "target_id", "similarity", "disease_priority",
                    "centrality", "weighted_score", "rank"
                ])
        
        return results
    
    def optimize_weights(
        self,
        validation_data: List[Tuple[str, str, float]],
        disease_id: Optional[str] = None,
        cv_folds: int = 5,
        grid_step: float = 0.1
    ) -> Dict[str, float]:
        """Optimize ranking weights using grid search."""
        # Compute node embeddings using stored node features if available
        if hasattr(self, 'node_features') and self.node_features is not None:
            # Use stored node features
            node_features = {k: v.to(self.device) for k, v in self.node_features.items()}
        else:
            # Convert knowledge graph to PyTorch Geometric format
            pyg_data = self.kg.to_torch_geometric()
        
            # Extract node features and move to device
            node_features = {}
            for entity_type in self.kg.entity_types:
                if entity_type in pyg_data and hasattr(pyg_data[entity_type], 'x'):
                    node_features[entity_type] = pyg_data[entity_type].x.to(self.device)
    
        # Extract edge indices and attributes
        edge_index_dict = {}
        edge_attr_dict = {}
    
        # Convert knowledge graph to PyTorch Geometric format
        pyg_data = self.kg.to_torch_geometric()
        for edge_type, edge_info in pyg_data.edge_items():
            edge_index_dict[edge_type] = edge_info.edge_index.to(self.device)
            edge_attr_dict[edge_type] = edge_info.edge_attr.to(self.device)
    
        # Compute node embeddings
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index_dict, edge_attr_dict)
    
        # Precompute scores for each compound-target pair
        precomputed_scores = {}
        for compound_id, target_id, _ in validation_data:
            if compound_id in self.kg.entity_to_idx.get("compound", {}) and target_id in self.kg.entity_to_idx.get("target", {}):
                if (compound_id, target_id) not in precomputed_scores:
                    # Calculate similarity score
                    try:
                        similarity = self.predict_score(compound_id, target_id, node_embeddings)
                    except Exception as e:
                        print(f"Warning: Could not compute similarity for {compound_id}-{target_id}: {e}")
                        similarity = 0.0
                
                    # Calculate disease priority
                    try:
                        disease_priority = self.calculate_disease_priority(target_id, disease_id)
                    except Exception as e:
                        print(f"Warning: Could not compute disease priority for {target_id}: {e}")
                        disease_priority = 0.0
                
                    # Calculate centrality
                    try:
                        centrality = self.calculate_centrality(target_id)
                    except Exception as e:
                        print(f"Warning: Could not compute centrality for {target_id}: {e}")
                        centrality = 0.0
                
                    # Store scores
                    precomputed_scores[(compound_id, target_id)] = {
                        "similarity": similarity,
                        "disease_priority": disease_priority,
                        "centrality": centrality
                    }
        
        # Define parameter grid
        weight_values = np.arange(0, 1 + grid_step, grid_step)
        param_grid = []
        
        for w1 in weight_values:
            for w2 in weight_values:
                w3 = 1 - w1 - w2
                if w3 >= 0 and w3 <= 1:
                    param_grid.append({
                        "similarity_weight": w1,
                        "disease_weight": w2,
                        "centrality_weight": w3
                    })
        
        # Define scoring function
        def score_weights(weights, data):
            y_true = []
            y_pred = []
            
            for compound_id, target_id, label in data:
                if (compound_id, target_id) in precomputed_scores:
                    scores = precomputed_scores[(compound_id, target_id)]
                    
                    # Calculate weighted score
                    weighted_score = (
                        weights["similarity_weight"] * scores["similarity"] +
                        weights["disease_weight"] * scores["disease_priority"] +
                        weights["centrality_weight"] * scores["centrality"]
                    )
                    
                    y_true.append(label)
                    y_pred.append(weighted_score)
            
            # Calculate ROC AUC
            if len(set(y_true)) < 2:
                return 0.0
            
            return roc_auc_score(y_true, y_pred)
        
        # Perform grid search with cross-validation
        best_score = 0.0
        best_weights = None
        
        # Split data into folds
        fold_size = len(validation_data) // cv_folds
        folds = []
        
        for i in range(cv_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < cv_folds - 1 else len(validation_data)
            folds.append(validation_data[start:end])
        
        # Evaluate each parameter combination
        for weights in param_grid:
            fold_scores = []
            
            for i in range(cv_folds):
                # Create train and validation splits
                val_fold = folds[i]
                train_folds = [fold for j, fold in enumerate(folds) if j != i]
                train_data = [item for fold in train_folds for item in fold]
                
                # Score weights
                fold_score = score_weights(weights, val_fold)
                fold_scores.append(fold_score)
            
            # Calculate average score
            avg_score = np.mean(fold_scores)
            
            # Update best weights
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights
        
        # Set weights to best values
        self.similarity_weight = best_weights["similarity_weight"]
        self.disease_weight = best_weights["disease_weight"]
        self.centrality_weight = best_weights["centrality_weight"]
        
        return best_weights
    
    def evaluate_ranking(
        self,
        test_data: List[Tuple[str, str, float]],
        disease_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate target ranking.
        
        Args:
            test_data: List of (compound_id, target_id, label) tuples.
            disease_id: Disease identifier. If None, disease priority is averaged.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Convert knowledge graph to PyTorch Geometric format
        pyg_data = self.kg.to_torch_geometric()
        
        # Extract node features and move to device
        node_features = {}
        for entity_type in self.kg.entity_types:
            if entity_type in pyg_data:
                node_features[entity_type] = pyg_data[entity_type].x.to(self.device)
        
        # Extract edge indices and attributes
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for edge_type, edge_info in pyg_data.edge_items():
            edge_index_dict[edge_type] = edge_info.edge_index.to(self.device)
            edge_attr_dict[edge_type] = edge_info.edge_attr.to(self.device)
        
        # Compute node embeddings
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index_dict, edge_attr_dict)
        
        # Initialize metrics
        y_true = []
        y_pred = []
        
        # Rank targets for each compound-target pair
        for compound_id, target_id, label in test_data:
            if (compound_id in self.kg.entity_to_idx["compound"] and
                target_id in self.kg.entity_to_idx["target"]):
                # Calculate similarity score
                similarity = self.predict_score(compound_id, target_id, node_embeddings)
                
                # Calculate disease priority
                disease_priority = self.calculate_disease_priority(target_id, disease_id)
                
                # Calculate centrality
                centrality = self.calculate_centrality(target_id)
                
                # Calculate weighted score
                weighted_score = (
                    self.similarity_weight * similarity +
                    self.disease_weight * disease_priority +
                    self.centrality_weight * centrality
                )
                
                y_true.append(label)
                y_pred.append(weighted_score)
        
        # Calculate metrics
        metrics = {}
        
        # ROC AUC
        if len(set(y_true)) >= 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        else:
            metrics["roc_auc"] = 0.0
        
        # Average precision
        metrics["avg_precision"] = average_precision_score(y_true, y_pred)
        
        # Mean rank
        ranks = []
        compound_groups = {}
        
        for i, (compound_id, target_id, label) in enumerate(test_data):
            if compound_id not in compound_groups:
                compound_groups[compound_id] = []
            
            compound_groups[compound_id].append(i)
        
        for compound_id, indices in compound_groups.items():
            if compound_id in self.kg.entity_to_idx["compound"]:
                # Get scores and labels for this compound
                compound_scores = [y_pred[i] for i in indices]
                compound_labels = [y_true[i] for i in indices]
                
                # Rank by scores
                sorted_indices = np.argsort(compound_scores)[::-1]
                
                # Find ranks of positive targets
                for i, idx in enumerate(sorted_indices):
                    if compound_labels[idx] == 1:
                        ranks.append(i + 1)
        
        # Mean rank
        if ranks:
            metrics["mean_rank"] = np.mean(ranks)
            metrics["median_rank"] = np.median(ranks)
        else:
            metrics["mean_rank"] = 0.0
            metrics["median_rank"] = 0.0
        
        # Mean reciprocal rank
        if ranks:
            metrics["mean_reciprocal_rank"] = np.mean([1.0 / r for r in ranks])
        else:
            metrics["mean_reciprocal_rank"] = 0.0
        
        return metrics



