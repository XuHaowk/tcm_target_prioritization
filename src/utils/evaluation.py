"""
Evaluation utilities for TCM target prioritization system.
"""
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.knowledge_graph import KnowledgeGraph
from src.models.attention_rgcn import AttentionRGCN
from src.ranking.target_ranker import TargetRanker

class Evaluator:
    """Evaluator for TCM target prioritization."""
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        model: AttentionRGCN,
        ranker: TargetRanker,
        device: str = "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            kg: Knowledge graph.
            model: AttentionRGCN model.
            ranker: Target ranker.
            device: Device to run on.
        """
        self.kg = kg
        self.model = model
        self.ranker = ranker
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_model(
        self,
        test_data: List[Tuple[str, str, float]],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of (compound_id, target_id, label) tuples.
            threshold: Classification threshold.
            
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
        y_score = []
        
        # Make predictions for each compound-target pair
        for compound_id, target_id, label in test_data:
            if (compound_id in self.kg.entity_to_idx["compound"] and
                target_id in self.kg.entity_to_idx["target"]):
                # Calculate similarity score
                similarity = self.ranker.predict_score(compound_id, target_id, node_embeddings)
                
                # Convert to binary prediction
                prediction = 1 if similarity >= threshold else 0
                
                y_true.append(label)
                y_pred.append(prediction)
                y_score.append(similarity)
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC
        if len(set(y_true)) >= 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        else:
            metrics["roc_auc"] = 0.0
        
        # Average precision
        metrics["avg_precision"] = average_precision_score(y_true, y_score)
        
        return metrics
    
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
        return self.ranker.evaluate_ranking(test_data, disease_id)
    
    def evaluate_compound_specific(
        self,
        compound_id: str,
        target_ids: List[str],
        labels: List[float],
        disease_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate ranking for a specific compound.
        
        Args:
            compound_id: Compound identifier.
            target_ids: List of target identifiers.
            labels: List of labels.
            disease_id: Disease identifier. If None, disease priority is averaged.
            top_k: Number of top targets to consider. If None, consider all.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Rank targets
        ranked_targets = self.ranker.rank_targets(
            compound_id=compound_id,
            disease_id=disease_id,
            top_k=None
        )
        
        # Create mapping from target ID to rank
        target_ranks = dict(zip(ranked_targets["target_id"], ranked_targets["rank"]))
        
        # Initialize metrics
        y_true = []
        y_rank = []
        
        # Get ranks for each target
        for target_id, label in zip(target_ids, labels):
            if target_id in target_ranks:
                y_true.append(label)
                y_rank.append(target_ranks[target_id])
        
        # Calculate metrics
        metrics = {}
        
        # Mean rank of positive targets
        positive_ranks = [rank for rank, label in zip(y_rank, y_true) if label == 1]
        if positive_ranks:
            metrics["mean_rank"] = np.mean(positive_ranks)
            metrics["median_rank"] = np.median(positive_ranks)
        else:
            metrics["mean_rank"] = 0.0
            metrics["median_rank"] = 0.0
        
        # Mean reciprocal rank
        if positive_ranks:
            metrics["mean_reciprocal_rank"] = np.mean([1.0 / r for r in positive_ranks])
        else:
            metrics["mean_reciprocal_rank"] = 0.0
        
        # Hit rate at top_k
        if top_k is not None and positive_ranks:
            metrics[f"hit_rate@{top_k}"] = sum(1 for r in positive_ranks if r <= top_k) / len(positive_ranks)
        else:
            metrics[f"hit_rate@{top_k if top_k is not None else 'all'}"] = 0.0
        
        return metrics
    
    def plot_precision_recall_curve(
        self,
        test_data: List[Tuple[str, str, float]],
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            test_data: List of (compound_id, target_id, label) tuples.
            output_file: Output file path. If None, display the plot.
        """
        # Evaluate model to get scores
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
        y_score = []
        
        # Make predictions for each compound-target pair
        for compound_id, target_id, label in test_data:
            if (compound_id in self.kg.entity_to_idx["compound"] and
                target_id in self.kg.entity_to_idx["target"]):
                # Calculate similarity score
                similarity = self.ranker.predict_score(compound_id, target_id, node_embeddings)
                
                y_true.append(label)
                y_score.append(similarity)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        # Calculate average precision
        avg_precision = average_precision_score(y_true, y_score)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def plot_target_rank_distribution(
        self,
        test_data: List[Tuple[str, str, float]],
        disease_id: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> None:
        """
        Plot distribution of target ranks.
        
        Args:
            test_data: List of (compound_id, target_id, label) tuples.
            disease_id: Disease identifier. If None, disease priority is averaged.
            output_file: Output file path. If None, display the plot.
        """
        # Group test data by compound
        compound_groups = {}
        for compound_id, target_id, label in test_data:
            if compound_id not in compound_groups:
                compound_groups[compound_id] = []
            
            compound_groups[compound_id].append((target_id, label))
        
        # Initialize rank lists
        positive_ranks = []
        negative_ranks = []
        
        # Process each compound
        for compound_id, targets in compound_groups.items():
            if compound_id in self.kg.entity_to_idx["compound"]:
                # Rank targets
                try:
                    ranked_targets = self.ranker.rank_targets(
                        compound_id=compound_id,
                        disease_id=disease_id,
                        top_k=None
                    )
                    
                    # Create mapping from target ID to rank
                    target_ranks = dict(zip(ranked_targets["target_id"], ranked_targets["rank"]))
                    
                    # Collect ranks
                    for target_id, label in targets:
                        if target_id in target_ranks:
                            rank = target_ranks[target_id]
                            if label == 1:
                                positive_ranks.append(rank)
                            else:
                                negative_ranks.append(rank)
                except Exception as e:
                    print(f"Error ranking targets for compound {compound_id}: {e}")
        
        # Plot rank distributions
        plt.figure(figsize=(12, 8))
        
        # Create bins
        max_rank = max(max(positive_ranks) if positive_ranks else 0,
                     max(negative_ranks) if negative_ranks else 0)
        bins = np.linspace(1, max_rank + 1, min(100, max_rank + 1))
        
        # Plot histograms
        plt.hist(positive_ranks, bins=bins, alpha=0.6, label="Positive", color="green", density=True)
        plt.hist(negative_ranks, bins=bins, alpha=0.6, label="Negative", color="red", density=True)
        
        plt.xlabel("Rank")
        plt.ylabel("Density")
        plt.title("Distribution of Target Ranks")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")
        
        # Save or display the plot
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def analyze_feature_importance(
        self,
        test_data: List[Tuple[str, str, float]],
        disease_id: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Analyze feature importance for ranking.
        
        Args:
            test_data: List of (compound_id, target_id, label) tuples.
            disease_id: Disease identifier. If None, disease priority is averaged.
            output_file: Output file path for the plot. If None, display the plot.
            
        Returns:
            Dictionary of feature importance scores.
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
        
        # Initialize feature values
        similarities = []
        disease_priorities = []
        centralities = []
        labels = []
        
        # Collect feature values
        for compound_id, target_id, label in test_data:
            if (compound_id in self.kg.entity_to_idx["compound"] and
                target_id in self.kg.entity_to_idx["target"]):
                # Calculate similarity score
                similarity = self.ranker.predict_score(compound_id, target_id, node_embeddings)
                
                # Calculate disease priority
                disease_priority = self.ranker.calculate_disease_priority(target_id, disease_id)
                # Calculate centrality
                centrality = self.ranker.calculate_centrality(target_id)
                
                # Store feature values
                similarities.append(similarity)
                disease_priorities.append(disease_priority)
                centralities.append(centrality)
                labels.append(label)
        
        # Convert to numpy arrays
        similarities = np.array(similarities)
        disease_priorities = np.array(disease_priorities)
        centralities = np.array(centralities)
        labels = np.array(labels)
        
        # Calculate correlation with labels
        similarity_corr = np.corrcoef(similarities, labels)[0, 1] if len(set(labels)) > 1 else 0
        disease_priority_corr = np.corrcoef(disease_priorities, labels)[0, 1] if len(set(labels)) > 1 else 0
        centrality_corr = np.corrcoef(centralities, labels)[0, 1] if len(set(labels)) > 1 else 0
        
        # Calculate feature importance based on correlation
        importance = {
            "similarity": abs(similarity_corr),
            "disease_priority": abs(disease_priority_corr),
            "centrality": abs(centrality_corr)
        }
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        # Create visualization
        if output_file is not None or output_file is None:  # Always create the plot
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            feature_names = list(importance.keys())
            importance_values = list(importance.values())
            
            plt.bar(feature_names, importance_values, color=["#3498db", "#2ecc71", "#e74c3c"])
            plt.ylabel("Normalized Importance")
            plt.title("Feature Importance for Target Ranking")
            plt.ylim([0, 1])
            
            # Add importance values as text
            for i, v in enumerate(importance_values):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
            
            # Save or display the plot
            if output_file is not None:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                plt.savefig(output_file)
                plt.close()
            else:
                plt.show()
        
        return importance
